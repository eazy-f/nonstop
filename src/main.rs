extern crate termion;
extern crate rand;
extern crate core;

use std::sync::mpsc::{self, Sender, Receiver};
use std::time::SystemTime;

use termion::color;
use termion::screen::*;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use std::string::String;
use core::ops::Index;
use std::iter::FromIterator;

use std::io::{Write, stdout, stdin};
use std::convert::From;
use std::fmt::Display;
use std::fmt;
use std::cmp::Ordering;
use std::thread;

type GroupSize = usize;
type Float32 = f32;

#[derive(Debug)]
enum UIMessage {
    UIQuit,
    UIUpdate,
    UIDataSourceUpdate(u32),
    UIKeyPress(termion::event::Key)
}

struct UIBox {
    x: u16,
    y: u16,
    width: u16,
    height: u16
}

struct VecGroup<T: Display + PartialOrd + PartialEq> {
    height: T,
    children: Option<Vec<VecGroup<T>>>
}

impl<T: Display + PartialOrd + PartialEq> fmt::Display for VecGroup<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.height)
    }
}

impl<T: Display + PartialOrd + PartialEq> PartialEq for VecGroup<T> {
    fn eq(&self, other: &Self) -> bool {
        self.height == other.height
    }
}

impl<T: Display + PartialOrd + PartialEq> Eq for VecGroup<T> {}

impl<T: Display + PartialOrd + PartialEq> PartialOrd for VecGroup<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.height.partial_cmp(&other.height)
    }
}

impl<T: Display + PartialOrd + PartialEq> Ord for VecGroup<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

trait Location: Clone + Send {
    fn distance(a: &Self, b: &Self) -> Float32;
}

/* Collection: Index<usize, Output=Group<T>>*/
trait Group<T: Display + PartialOrd + PartialEq> {
    type Collection: IntoIterator;
    /*type ChildrenIter: Iterator;*/
    fn height(self) -> T;
    fn children(self) -> Option<Self::Collection>;
    fn len(self) -> GroupSize;
    /* FIXME: no need for a trait object */
    fn iter<'a>(&'a self) -> Box<Iterator<Item=&Self> + 'a>;
}

impl<T: Display + PartialOrd + PartialEq> Group<T> for VecGroup<T> {
    type Collection = Vec<VecGroup<T>>;
    /*type ChildrenIter = Iter<'a, VecGroup<T>>;*/
    fn height(self) -> T {
        self.height
    }
    fn children(self) -> Option<Self::Collection> {
        self.children
    }
    fn len(self) -> GroupSize {
        self.children.map_or(0, |v| v.len())
    }
    fn iter<'a>(&'a self) -> Box<Iterator<Item=&Self> + 'a> {
        let empty = Vec::new();
        let children: &Option<Vec<VecGroup<T>>> = &self.children;
        match children {
            Some(ref c) => Box::new(c.into_iter()),
            None => Box::new(empty.into_iter())
        }
    }
}

trait IntoGroup<T: Display + PartialOrd + PartialEq> {
    type GroupType: Group<T>;
    fn group_avg(self, groups: GroupSize) -> Self::GroupType;
}

impl<T, S> IntoGroup<Float32> for S
    where T: Location,
          S: Segment<T, Item=Position<T>, Output=Position<T>>
{
    type GroupType = VecGroup<Float32>;
    fn group_avg(self, groups: GroupSize) -> Self::GroupType {
        let mut last: Option<Position<T>> = None;
        let mut distances = Vec::new();
        for current in self.into_iter() {
            for prev in last.iter() {
                distances.push(Location::distance(&prev.location, &current.location));
            }
            last = Some(current);
        }
        let leaf_group = |height: &Float32| VecGroup{height: *height, children: None};
        let height = distances.iter().fold((0.0, 0 as usize), average_folder).0;
        let children = Vec::from_iter(distances[0..groups].into_iter().map(leaf_group));
        VecGroup {
            height: height,
            children: Some(children)
        }
    }
}

/*fn average_folder<T>((current, count): (T, usize), sample: T) -> (T, usize)
    where T: Div<Output=T> + Add + Mul + From<usize> {*/
fn average_folder((current, count): (Float32, usize), sample: &Float32)
                  -> (Float32, usize) {
    let new_count = count + 1;
    let avg = match count {
        0 => *sample,
        _ => (f32::from(new_count as u16))*(current + (sample / f32::from(count as u16)))
    };
    (avg, new_count)
}

impl Location for u32 {
    fn distance(a: &Self, b: &Self) -> Float32 {
        let a = ((*b as i64 - *a as i64) % 1024) as Float32;
        a
    }
}

#[derive(Clone)]
struct Position<T: Location> {
    time: SystemTime,
    location: T
}

trait Segment<T: Location>: Index<usize> + IntoIterator + Clone + Send {
    fn name(&self) -> String;
}

impl<T: Location> Segment<T> for Vec<Position<T>> {
    fn name(&self) -> String {
        String::from("no name")
    }
}

fn window_init() -> AlternateScreen<termion::raw::RawTerminal<std::io::Stdout>> {
    let mut screen = AlternateScreen::from(stdout().into_raw_mode().unwrap());
    write!(screen, "{}", termion::clear::All);
    write!(screen, "{}{}Loading tracks", termion::cursor::Goto(1, 1), color::Fg(color::Red));
    screen.flush().unwrap();
    screen
}

fn load_test_tracks(segments: Sender<Vec<Position<u32>>>) {
    let size = 50000;
    let mut positions = Vec::new();
    for _ in 0..size {
        let position = Position{
            location: rand::random(),
            time: SystemTime::now()
        };
        positions.push(position);
    }
    segments.send(positions);
}

fn show_segment<W: Write, L: Location, T: Segment<L>>(screen: &mut AlternateScreen<W>, segment: &T) {
    write!(screen, "{}", &segment.name());
    screen.flush().unwrap();
}

fn window_run<L, T, W>(segments: Receiver<T>, screen: &mut AlternateScreen<W>)
    where L: Location, T: Segment<L, Item=Position<L>, Output=Position<L>> + 'static, W: Write
{
    let (ui_tx, ui_rx) = mpsc::channel();
    let beacon = 12; /* FIXME: such an ugly hack */
    let mut elements = initial_ui_elements(&ui_tx, segments);
    ui_tx.send(UIMessage::UIUpdate);
    write!(screen, "{}", termion::clear::All);
    for message in ui_rx.iter() {
        match message {
            UIMessage::UIUpdate => {
                elements.iter().for_each(|x| x.draw(screen));
                screen.flush().unwrap();
            },
            UIMessage::UIDataSourceUpdate(_) |
            UIMessage::UIKeyPress(_) => {
                let mut stuff: Vec<Box<UIElement<W>>> = elements.iter_mut().filter_map(|x| x.update(&message, &beacon)).collect();
                if stuff.len() > 0 {
                    ui_tx.send(UIMessage::UIUpdate);
                }
                elements.append(&mut stuff);
            },
            UIMessage::UIQuit => break
        }
    }
}

trait UIElement<W: Write> {
    fn draw(&self, screen: &mut AlternateScreen<W>) {}
    fn update<'a>(&mut self, message: &UIMessage, beacon: &'a u32) -> Option<Box<UIElement<W> + 'a>> {None}
}

struct BarWindow {
    heights: Vec<f32>,
    ui_box: UIBox
}

struct ElementQuit<'a> {
    ui_events: &'a Sender<UIMessage>
}

struct ElementSegmentSelector<'a, T, L, W>
    where L: Location, T: Segment<L, Item=Position<L>, Output=Position<L>>, W: Write
{
    ui_events: &'a Sender<UIMessage>,
    ui_box: UIBox,
    available_segments: Vec<T>,
    segments: Receiver<T>,
    disconnected: bool,
    data_source_id: u32,
    type_trick: Option<W>
}

impl BarWindow {
    fn new<T: Iterator<Item=Float32>>(from: T, ui_box: UIBox) -> BarWindow {
        BarWindow{heights: from.map(|x| x.max(0.0).min(1.0)).collect(), ui_box: ui_box}
    }
}

impl<W: Write> UIElement<W> for BarWindow {
    fn draw(&self, screen: &mut AlternateScreen<W>) {
        let ui_box = &self.ui_box;
        let win_width = self.heights.len().min(ui_box.width as usize);
        let height = ui_box.height;
        let heights: Vec<u16> = self.heights[0..win_width].iter().map(|x| (x * (height as f32)) as u16).collect();
        for i in 0..height {
            write!(screen, "{}", termion::cursor::Goto(ui_box.x, ui_box.y + (height - i)));
            let line: String = heights.iter().map(|h| if *h < (i+1) {' '} else {'*'}).collect();
            write!(screen, "{}", line);
        }
    }
}

impl<'a> ElementQuit<'a> {
    fn new(ui_events: &'a Sender<UIMessage>) -> ElementQuit<'a> {
        let worker_ui_events = ui_events.clone();
        thread::spawn(move || {
            for c in stdin().keys() {
                worker_ui_events.send(UIMessage::UIKeyPress(c.unwrap()));
            }
        });
        ElementQuit{ui_events: ui_events}
    }
}

impl<'a, W: Write> UIElement<W> for ElementQuit<'a> {
    fn update<'b>(&mut self, message: &UIMessage, _beacon: &'b u32) -> Option<Box<UIElement<W> + 'b>> {
        match message {
            UIMessage::UIKeyPress(Key::Char('q')) => {self.ui_events.send(UIMessage::UIQuit);},
            _ => ()
        };
        None
    }
}

impl<'a, T, L, W> ElementSegmentSelector<'a, T, L, W>
    where L: Location, T: Segment<L, Item=Position<L>, Output=Position<L>> + 'static, W: Write
{
    fn new(segments: Receiver<T>, ui_events: &'a Sender<UIMessage>, ui_box: UIBox) -> ElementSegmentSelector<'a, T, L, W> {
        let (segments_tx, segments_rx) = mpsc::channel();
        let data_source_id = 1;
        let worker_ui_events = ui_events.clone();
        thread::spawn(move || dispatch_segments(segments, segments_tx, worker_ui_events, data_source_id));
        ElementSegmentSelector{
            ui_events: ui_events,
            ui_box: ui_box,
            available_segments: Vec::new(),
            segments: segments_rx,
            disconnected: false,
            data_source_id: data_source_id,
            type_trick: None
        }
    }

    fn data_source_update(&mut self) -> Option<Box<UIElement<W>>> {
        while true {
            match self.segments.try_recv() {
                Err(err) => {
                    if err == std::sync::mpsc::TryRecvError::Disconnected {
                        self.disconnected = true
                    }
                    break
                }
                Ok(segment) => {
                    self.available_segments.push(segment);
                }
            }
        }
        match (self.disconnected, self.available_segments.len()) {
            (true, 0) => {
                self.ui_events.send(UIMessage::UIQuit);
                None
            },
            (true, 1) => Some(Box::new(state_segment_edit(self.available_segments[0].clone()))),
            _ => {
                self.ui_events.send(UIMessage::UIUpdate);
                None
            }
        }
    }
}

impl<'a, T, L, W> UIElement<W> for ElementSegmentSelector<'a, T, L, W>
    where L: Location, T: Segment<L, Item=Position<L>, Output=Position<L>> + 'static, W: Write
{
    fn update<'b>(&mut self, message: &UIMessage, _beacon: &'b u32) -> Option<Box<UIElement<W> + 'b>> {
        match message {
            UIMessage::UIDataSourceUpdate(id) if *id == self.data_source_id => self.data_source_update(),
            _ => None
        }
    }
}

fn dispatch_segments<L, T>(from: Receiver<T>, to: Sender<T>, ui_events: Sender<UIMessage>, data_source_id: u32)
    where L: Location, T: Segment<L, Item=Position<L>, Output=Position<L>>
{
    for segment in from.iter() {
        to.send(segment);
        ui_events.send(UIMessage::UIDataSourceUpdate(data_source_id));
    }
}

fn state_segment_edit<L, T>(segment: T) -> BarWindow
    where L: Location, T: Segment<L, Item=Position<L>, Output=Position<L>>
{
    let bars = 40;
    let group = segment.group_avg(bars as GroupSize);
    let max = group.iter().max().unwrap().height;
    let normalized = group.iter().map(|x| x.height / max);
    BarWindow::new(normalized, whole_window())
}

fn initial_ui_elements<'a, L, T, W>(ui_events: &'a Sender<UIMessage>, segments: Receiver<T>) -> Vec<Box<UIElement<W> + 'a>>
    where L: Location + 'a, T: Segment<L, Item=Position<L>, Output=Position<L>> + 'static, W: Write + 'a
{
    vec![Box::new(ElementQuit::new(ui_events)),
         Box::new(ElementSegmentSelector::new(segments, ui_events, whole_window()))]
}

fn whole_window() -> UIBox {
    UIBox{x: 1, y: 1, width: 40, height: 20}
}

fn main() {
    let (tx, rx) = mpsc::channel();
    let mut screen = window_init();
    load_test_tracks(tx);
    window_run(rx, &mut screen);
}
