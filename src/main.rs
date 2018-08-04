extern crate termion;
extern crate rand;
extern crate core;
extern crate quick_xml;
extern crate chrono;

pub mod xml;
pub mod segment;

use std::sync::mpsc::{self, Sender, Receiver};
use std::env;

use termion::color;
use termion::screen::*;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use std::string::String;

use std::io::{Write, stdout, stdin};
use std::convert::From;
use std::fmt::Display;
use std::fmt;
use std::cmp::Ordering;
use std::thread;
use std::iter;
use core::ops::Index;

use chrono::prelude::*;

use segment::{Segment, Location, Position, Float32};

type GroupSize = usize;
type GroupIndex = GroupSize;

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

/* Collection: Index<usize, Output=Group<T>>*/
trait Group<T: Display + PartialOrd + PartialEq>: Ord {
    type Collection: IntoIterator + Index<GroupIndex, Output=Self>;
    /*type ChildrenIter: Iterator;*/
    fn height(&self) -> &T;
    fn children(&self) -> Option<&Self::Collection>;
    fn len(self) -> GroupSize;
    /* FIXME: no need for a trait object */
    fn iter<'a>(&'a self) -> Box<Iterator<Item=&Self> + 'a>;
}

impl<T: Display + PartialOrd + PartialEq> Group<T> for VecGroup<T> {
    type Collection = Vec<VecGroup<T>>;
    /*type ChildrenIter = Iter<'a, VecGroup<T>>;*/
    fn height(&self) -> &T {
        &self.height
    }
    fn children(&self) -> Option<&Self::Collection> {
        self.children.as_ref()
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
        let mut speed = Vec::new();
        for current in self.into_iter() {
            for prev in last.iter() {
                let distance = Location::distance(&prev.location, &current.location);
                let time = current.time.signed_duration_since(prev.time).num_seconds() as f32;
                speed.push(distance / time);
            }
            last = Some(current);
        }
        let leaf_group = |height: &Float32| VecGroup{height: *height, children: None};
        build_distances_group(speed.iter().map(leaf_group).collect(), groups)
    }

}

fn build_distances_group(groups: Vec<VecGroup<Float32>>, limit: GroupSize) -> VecGroup<Float32> {
    let min_size = groups.len() / limit;
    let reminder = groups.len() % limit;
    let mut chunks = vec![Box::new(Vec::new())]; /* FIXME: use arrays - size is known */
    groups.into_iter().enumerate().for_each(|(i, group)| {
        let group_limit = if chunks.len() <= reminder {
            min_size + 1
        } else {
            min_size
        };
        if chunks.last().unwrap().len() < group_limit {
            chunks.last_mut().unwrap().push(group)
        } else {
            chunks.push(Box::new(vec!(group)))
        }
    });
    let children = if chunks.len() < limit {
        chunks.into_iter().map(|c| c.into_iter().fold(None, |_, g| Some(g)).unwrap()).collect()
    } else {
        chunks.into_iter().map(|groups| build_distances_group(*groups, limit)).collect()
    };
    build_supergroup(children)
}

fn build_supergroup(groups: Vec<VecGroup<Float32>>) -> VecGroup<Float32> {
    let height = groups.iter().map(|g| *g.height()).fold((0.0, 0 as usize), average_folder).0;
    VecGroup {
        height: height,
        children: Some(groups)
    }
}


/*fn average_folder<T>((current, count): (T, usize), sample: T) -> (T, usize)
    where T: Div<Output=T> + Add + Mul + From<usize> {*/
fn average_folder((current, count): (Float32, usize), sample: Float32)
                  -> (Float32, usize) {
    let new_count = count + 1;
    let avg = (f32::from(count as u16)*current + sample) / f32::from(new_count as u16);
    (avg, new_count)
}

impl Location for u32 {
    fn distance(a: &Self, b: &Self) -> Float32 {
        let a = ((*b as i64 - *a as i64) % 1024) as Float32;
        if a > 0.0 {
            a
        } else {
            -a
        }
    }
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
            time: Utc::now()
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

struct BarWindow<T: Group<Float32>> {
    heights: Option<Vec<f32>>,
    ui_box: UIBox,
    ui_events: Sender<UIMessage>,
    selected: Option<GroupSize>,
    levels: Vec<GroupIndex>,
    group: T
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

impl<T: Group<Float32>> BarWindow<T> {
    fn new(group: T, ui_events: Sender<UIMessage>, ui_box: UIBox) -> BarWindow<T>
    {
        BarWindow{
            heights: None,
            ui_box: ui_box,
            ui_events: ui_events,
            selected: None,
            group: group,
            levels: Vec::new()
        }
    }

    fn key_pressed(&mut self, key: Key) {
        let heights = self.calculate_heights();
        let width = heights.len();
        let center = width / 2;
        let curent_pos = self.selected.unwrap_or(center);
        let min = 0;
        let max = width - 1;
        let new_pos = match key {
            Key::Left | Key::Char('h') if curent_pos > min =>
                Some(curent_pos - 1),
            Key::Right | Key::Char('l') if curent_pos < max =>
                Some(curent_pos + 1),
            Key::Char('\n') => {
                let selected = self.selected; /* FIXME: make a oneliner */
                selected.iter().for_each(|x| self.select_subgroup(*x));
                None
            },
            Key::Char('u') => self.select_supergroup(),
            _ => self.selected
        };
        self.selected = new_pos;
        self.ui_events.send(UIMessage::UIUpdate);
    }

    fn select_subgroup(&mut self, subgroup: GroupIndex) {
        self.heights = None;
        self.levels.push(subgroup);
    }

    fn select_supergroup(&mut self) -> Option<GroupIndex> {
        self.heights = None;
        self.levels.pop()
    }

    fn calculate_heights(&self) -> Vec<Float32> {
        let group = self.levels.iter().fold(&self.group, group_level_walker);
        let max = group.iter().max().unwrap().height();
        let normalize = |x: &T| -> Float32 {
            (x.height() / max).max(0.0).min(1.0)
        };
        group.iter().map(normalize).collect()
    }
}

fn group_level_walker<'a, T>(group: &'a T, i: &GroupIndex) -> &'a T
    where T: Group<Float32>
{
    let go_deeper = |children: &'a T::Collection| -> &'a T {
        let child = &children[*i];
        if child.children().is_some() {
            child
        } else {
            group
        }
    };
    group.children().map_or(group, go_deeper)
}


impl<T, W> UIElement<W> for BarWindow<T>
    where W: Write, T: Group<Float32>
{
    fn draw(&self, screen: &mut AlternateScreen<W>) {
        let ui_box = &self.ui_box;
        let float_heights = self.calculate_heights();
        let win_width = float_heights.len().min(ui_box.width as usize);
        let height = ui_box.height;
        let heights: Vec<u16> = float_heights[0..win_width].iter().map(|x| (x * (height as f32)) as u16).collect();
        let len = heights.len();
        for i in 0..height {
            write!(screen, "{}", termion::cursor::Goto(ui_box.x, ui_box.y + (height - i)));
            write!(screen, "{}", color::Fg(color::Red));
            let filled = '*';
            let empty = ' ';
            let full_heights = heights.iter().chain(iter::repeat(&0).take((ui_box.width as usize) - len));
            let line: String = full_heights.map(|y| if *y < (i+1) {empty} else {filled}).collect();
            write!(screen, "{}", line);
            let selector = |x: &usize| {
                let pos = termion::cursor::Goto(*x as u16 + ui_box.x, ui_box.y + (height - i));
                if heights[*x] > i {
                    write!(screen, "{}{}{}", pos, color::Fg(color::Blue), filled);
                }
            };
            self.selected.iter().for_each(selector);
        }
    }

    fn update<'b>(&mut self, message: &UIMessage, _beacon: &'b u32) -> Option<Box<UIElement<W> + 'b>> {
        match message {
            UIMessage::UIKeyPress(key)
                if *key == Key::Left
                || *key == Key::Right
                || *key == Key::Char('\n')
                || *key == Key::Char('u')
                || *key == Key::Char('h')
                || *key == Key::Char('l') => self.key_pressed(*key),
            _ => ()
        };
        None
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
        loop {
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
            (true, 1) => Some(Box::new(state_segment_edit(self.available_segments[0].clone(), self.ui_events))),
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

fn state_segment_edit<L, T>(segment: T, ui_events: &Sender<UIMessage>) -> BarWindow<T::GroupType>
    where L: Location,
          T: Segment<L, Item=Position<L>, Output=Position<L>> + IntoGroup<Float32>,
{
    let group = segment.group_avg(40 as GroupSize);
    BarWindow::new(group, ui_events.clone(), whole_window())
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
    let args: Vec<String> = env::args().collect();
    let (tx, rx) = mpsc::channel();
    let mut screen = window_init();
    /*load_test_tracks(tx);*/
    xml::segments_from_file(&args[1], tx);
    window_run(rx, &mut screen);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn average() {
        let values = vec![-2.5, 1.0, 4.5];
        assert_eq!(values.into_iter().fold((0.0, 0 as usize), average_folder), (1.0, 3));
    }
}
