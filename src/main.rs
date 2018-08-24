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
use chrono::Duration;

use segment::{Segment, Location, Position, Float32};

type GroupSize = usize;
type GroupIndex = GroupSize;
type Bounds = (Duration, Duration);

#[derive(Debug, Clone)]
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
    duration: Duration,
    children: Option<Vec<VecGroup<T>>>
}

impl<T: Display + PartialOrd + PartialEq> VecGroup<T> {
    fn split_at(self, split_point: Duration) -> (Option<Self>, Option<Self>) {
        if self.duration < split_point {
            (Some(self), None)
        } else {
            (None, Some(self))
        }
    }
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
    fn duration(&self) -> &Duration;
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
    fn duration(&self) -> &Duration {
        &self.duration
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
    fn group_avg(&self, groups: GroupSize) -> Self::GroupType;
}

impl<T, S> IntoGroup<Float32> for S
    where T: Location,
          S: Segment<T, Item=Position<T>, Output=Position<T>>,
          for<'a> &'a S: IntoIterator<Item = &'a Position<T>>
{
    type GroupType = VecGroup<Float32>;
    fn group_avg(&self, groups: GroupSize) -> Self::GroupType {
        let mut last: Option<&Position<T>> = None;
        let mut speed = Vec::new();
        for current in self.into_iter() {
            for prev in last.iter() {
                let distance = Location::distance(&prev.location, &current.location);
                /* FIXME: dirty workaround for negative time */
                let time = current.time.signed_duration_since(prev.time).max(Duration::milliseconds(1));
                speed.push((distance * 3600.0 / (time.num_milliseconds() as f32), time));
            }
            last = Some(current);
        }
        let leaf_group = |(height, duration): (Float32, Duration)| {
            VecGroup{height: height,
                     duration: duration,
                     children: None}
        };
        let start = Duration::seconds(0);
        let span = speed.iter().fold(start, |acc, (_, d)| acc + *d);
        build_speed_group(speed.into_iter().map(leaf_group).collect(), groups, span)
    }
}

fn build_speed_group(groups: Vec<VecGroup<Float32>>,
                     limit: GroupSize,
                     span: Duration) -> VecGroup<Float32>
{
    if groups.len() == 1 {
        groups.into_iter().last().unwrap()
    } else {
        let subgroup_span = span / (limit as i32);
        let mut splitted = vec![Box::new(Vec::new())];
        let mut work_span = subgroup_span;
        for group in groups {
            let (left, right) = group.split_at(work_span);
            left.map(|left_group| {
                work_span = work_span - left_group.duration;
                splitted.last_mut().unwrap().push(left_group)
            });
            right.map(|right_group| {
                splitted.push(Box::new(vec![right_group]));
                work_span = subgroup_span;
            });
        }
        let children = splitted.into_iter().filter_map(|subgroups| {
            if subgroups.len() > 0 {
                Some(build_speed_group(*subgroups, limit, subgroup_span))
            } else {
                None
            }
        }).collect();
        build_supergroup(children)
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
            chunks.push(Box::new(vec![group]))
        }
    });
    let children = if chunks.len() < limit {
        chunks.into_iter().map(|c| c.into_iter().fold(None, |_, g| Some(g)).unwrap()).collect()
    } else {
        chunks.into_iter().map(|groups| build_distances_group(*groups, limit)).collect()
    };
    build_supergroup(children)
}

fn build_supergroup(groups: Vec<VecGroup<Float32>>) ->
    VecGroup<Float32>
{
    let zero = (0.0, Duration::seconds(0));
    let (height, duration) = groups.iter().map(|g| (*g.height(), *g.duration())).fold(zero, average_folder);
    VecGroup {
        height: height,
        duration: duration,
        children: Some(groups)
    }
}


/*fn average_folder<T>((current, count): (T, usize), sample: T) -> (T, usize)
    where T: Div<Output=T> + Add + Mul + From<usize> {*/
fn average_folder((current, duration): (Float32, Duration),
                  (sample, sample_duration): (Float32, Duration))
                  -> (Float32, Duration) {
    let new_duration = duration + sample_duration;
    let milli = |duration: Duration| duration.num_milliseconds() as f32;
    let avg = (milli(duration) * current + milli(sample_duration) * sample) / milli(new_duration);
    (avg, new_duration)
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
        let name = |pos: &Position<T>| format!("{} {} points", pos.time, self.len());
        self.get(0).map_or(String::from("empty segment"), name)
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
    where L: Location + 'static,
          T: Segment<L, Item=Position<L>, Output=Position<L>> + IntoGroup<Float32> + 'static,
          W: Write + 'static
{
    let (ui_tx, ui_rx) = mpsc::channel();
    let mut loaded_segments = Vec::new();
    let lol = &mut loaded_segments;
    let beacon = 12; /* FIXME: such an ugly hack */
    {
        let mut elements = initial_ui_elements(ui_tx.clone(), segments, lol);
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
                let mut stuff = Vec::new();
                {
                    let lol = &mut stuff;
                    elements.iter_mut().for_each(|x| {
                        x.update(&message).map(|e| lol.push(e));
                    });
                }
/*                if stuff.len() > 0 {
                    ui_tx.send(UIMessage::UIUpdate);
            }*/
                elements.append(&mut stuff);
            },
            UIMessage::UIQuit => break
        }
    }
    }
}

trait Cutter {
    fn mark(&mut self, selected: Vec<GroupIndex>);
    fn is_cut(&self, candidate: &Vec<GroupIndex>) -> bool;
}

struct FatalityCutter {
    limits: Option<(Vec<GroupIndex>, Option<Vec<GroupIndex>>)>
}

impl FatalityCutter {
    fn new() -> FatalityCutter {
        FatalityCutter {limits: None}
    }
}

impl Cutter for FatalityCutter {
    fn mark(&mut self, selected: Vec<GroupIndex>) {
        self.limits = match self.limits {
            Some((ref left, None)) if left < &selected => {
                let mut right = selected.clone();
                let last = right.len() - 1;
                right[last] += 1;
                Some((left.clone(), Some(right)))
            },
            Some((ref left, None)) if left >= &selected => {
                let mut right = left.clone();
                let last = right.len() - 1;
                right[last] += 1;
                Some((selected, Some(right)))
            },
            _ => Some((selected, None))
        }
    }
    fn is_cut(&self, candidate: &Vec<GroupIndex>) -> bool {
        match self.limits {
            Some((ref left, Some(ref right))) if candidate >= left && candidate < right => true,
            _ => false
        }
    }
}

trait UIElement<W: Write> {
    fn draw(&self, screen: &mut AlternateScreen<W>) {}
    fn update<'a>(&'a mut self, message: &UIMessage) -> Option<Box<UIElement<W> + 'a>> {None}
}

struct BarWindow<T: Group<Float32>, C: Cutter, S: Copy> {
    heights: Option<Vec<f32>>,
    ui_box: UIBox,
    ui_events: Sender<UIMessage>,
    selected: Option<GroupSize>,
    levels: Vec<GroupIndex>,
    group: T,
    cutter: C,
    source: S
}

struct ElementQuit {
    ui_events: Sender<UIMessage>
}

struct ElementSegmentSelector<'a, T, L, W>
    where L: Location + 'a,
          T: Segment<L, Item=Position<L>, Output=Position<L>> + 'a,
          W: Write
{
    ui_events: Sender<UIMessage>,
    ui_box: UIBox,
    available_segments: &'a mut Vec<T>,
    segments: Receiver<T>,
    disconnected: bool,
    data_source_id: u32,
    type_trick: Option<W>,
    selected: Option<usize>
}

impl<T: Group<Float32>, C: Cutter, S: Copy> BarWindow<T, C, S> {
    fn new(source: S, into_group: fn(S) -> T, ui_events: Sender<UIMessage>,
           ui_box: UIBox, cutter: C) -> BarWindow<T, C, S>
    {
        BarWindow {
            heights: None,
            ui_box: ui_box,
            ui_events: ui_events,
            selected: None,
            group: into_group(source),
            levels: Vec::new(),
            cutter: cutter,
            source: source
        }
    }

    fn key_pressed(&mut self, key: Key) {
        let heights = calculate_heights(self.selected_group());
        let width = heights.len();
        let center = width / 2;
        let curent_pos = self.selected.unwrap_or(center);
        let min = 0;
        let max = width - 1;
        if key == Key::Char(' ') {
            self.selected.map(|group| {
                let mut levels = self.levels.clone();
                levels.push(group);
                self.cutter.mark(levels)
            });
        }
        if key == Key::Char('x') {
            self.selected.map(|group| {
                let mut levels = self.levels.clone();
                levels.push(group);
                self.cutter.mark(levels)
            });
        }
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

    fn selected_group(&self) -> &T {
        self.levels.iter().fold(&self.group, group_level_walker)
    }
}

fn calculate_heights<T: Group<Float32>>(group: &T) -> Vec<Float32> {
    let max = max_height(group);
    let normalize = |x: &T| -> Float32 {
        (x.height() / max).max(0.0).min(1.0)
    };
    group.iter().map(normalize).collect()
}

fn max_height<T: Group<Float32>>(group: &T) -> Float32 {
    group.iter().max().unwrap().height().clone()
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

#[derive(PartialEq, Clone, Copy)]
enum BarColors {
    Normal,
    Pointer,
    Selected
}

impl<T, W, C, S> UIElement<W> for BarWindow<T, C, S>
    where W: Write, T: Group<Float32>, C: Cutter, S: Copy
{
    fn draw(&self, screen: &mut AlternateScreen<W>) {
        let ui_box = &self.ui_box;
        let group = self.selected_group();
        let float_heights = calculate_heights(group);
        let max_height = max_height(group).to_string();
        let legend_width = max_height.len() as u16;
        let graph_box = UIBox {y: ui_box.y + 1, height: ui_box.height - 2, ..*ui_box};
        let bars = float_heights.len().min(ui_box.width as usize);
        let height = graph_box.height;
        let heights: Vec<u16> = float_heights[0..bars].iter().map(|x| (x * (height as f32)) as u16).collect();
        let len = heights.len();
        let mut levels = self.levels.clone();
        levels.push(0);
        for i in 0..(ui_box.width as usize) {
            let mut active_color = None;
            let filled = '*';
            let empty = ' ';
            levels[self.levels.len()] = i;
            for j in 0..height {
                let pos = termion::cursor::Goto(graph_box.x + i as u16, graph_box.y + (height - (j as u16) - 1));
                write!(screen, "{}", pos);
                let color = match (self.selected, self.cutter.is_cut(&levels)) {
                    (Some(bar), _) if bar == i => BarColors::Pointer,
                    (_, true) => BarColors::Selected,
                    _ => BarColors::Normal
                };
                if active_color != Some(color) {
                    /* FIXME: try solving with macro */
                    match color {
                        BarColors::Normal   => write!(screen, "{}", color::Fg(color::Green)),
                        BarColors::Pointer  => write!(screen, "{}", color::Fg(color::Blue)),
                        BarColors::Selected => write!(screen, "{}", color::Fg(color::Red))
                    };
                    active_color = Some(color);
                }
                let sym = if i < heights.len() && j < heights[i] {
                    filled
                } else {
                    empty
                };
                write!(screen, "{}", sym);
            }
        }
        draw_time(screen, ui_box, &group.duration());
        self.selected.iter().for_each(|pos| draw_speed(screen, &ui_box, pos, group.children().unwrap()[*pos].height()));
    }

    fn update<'a>(&'a mut self, message: &UIMessage) -> Option<Box<UIElement<W> + 'a>> {
        match message {
            UIMessage::UIKeyPress(key)
                if *key == Key::Left
                || *key == Key::Right
                || *key == Key::Char('\n')
                || *key == Key::Char(' ')
                || *key == Key::Char('u')
                || *key == Key::Char('h')
                || *key == Key::Char('l') => self.key_pressed(*key),
            _ => ()
        };
        None
    }
}

fn draw_time<W: Write>(screen: &mut AlternateScreen<W>, ui_box: &UIBox, time: &Duration) {
    let seconds_total = time.num_seconds();
    let seconds = seconds_total % 60;
    let minutes = (seconds_total / 60) % 60;
    let hours = seconds_total / (60 * 60);
    let displayed = format!("{}:{:02}:{:02}", hours, minutes, seconds);
    let start_x = ui_box.x + (ui_box.width - displayed.len() as u16) / 2;
    let first_line = ui_box.y;
    write!(screen, "{}", termion::cursor::Goto(start_x, first_line));
    write!(screen, "{}", color::Fg(color::Yellow));
    write!(screen, "{}", displayed);
}

fn draw_speed<W: Write>(screen: &mut AlternateScreen<W>, ui_box: &UIBox, pos: &GroupIndex, speed: &Float32) {
    let displayed = format!("{}", speed);
    let last_line = ui_box.y + ui_box.height - 1;
    let displayed_len = displayed.len() as u16;
    let left_len = displayed_len / 2;
    let right_border = ui_box.width - 1 - displayed_len;
    let left_shift = ((*pos as i32 - (left_len as i32)).max(0) as u16).min(right_border);
    let space = ' ';
    let left_pad: String = iter::repeat(space).take(left_shift as usize).collect();
    let right_pad: String = iter::repeat(space).take((ui_box.width - (left_shift + displayed_len)) as usize).collect();
    write!(screen, "{}", termion::cursor::Goto(ui_box.x, last_line));
    write!(screen, "{}", color::Fg(color::Blue));
    write!(screen, "{}{}{}", left_pad, displayed, right_pad);
}

impl ElementQuit {
    fn new(ui_events: Sender<UIMessage>) -> ElementQuit {
        let worker_ui_events = ui_events.clone();
        thread::spawn(move || {
            for c in stdin().keys() {
                worker_ui_events.send(UIMessage::UIKeyPress(c.unwrap()));
            }
        });
        ElementQuit{ui_events: ui_events}
    }
}

impl<'a, W: Write> UIElement<W> for ElementQuit {
    fn update<'b>(&'b mut self, message: &UIMessage) -> Option<Box<UIElement<W> + 'b>> {
        match message {
            UIMessage::UIKeyPress(Key::Char('q')) => {self.ui_events.send(UIMessage::UIQuit);},
            _ => ()
        };
        None
    }
}

impl<'a, T, L, W> ElementSegmentSelector<'a, T, L, W>
    where L: Location,
          T: Segment<L, Item=Position<L>, Output=Position<L>> + IntoGroup<Float32> + 'static,
          W: Write
{
    fn new(segments: Receiver<T>,
           loaded_segments: &'a mut Vec<T>,
           ui_events: Sender<UIMessage>,
           ui_box: UIBox) -> ElementSegmentSelector<'a, T, L, W>
    {
        let (segments_tx, segments_rx) = mpsc::channel();
        let data_source_id = 1;
        let worker_ui_events = ui_events.clone();
        thread::spawn(move || dispatch_segments(segments, segments_tx, worker_ui_events, data_source_id));
        ElementSegmentSelector{
            ui_events: ui_events,
            ui_box: ui_box,
            segments: segments_rx,
            disconnected: false,
            data_source_id: data_source_id,
            type_trick: None,
            selected: None,
            available_segments: loaded_segments
        }
    }

    fn data_source_update<'b>(&'b mut self) -> Option<Box<UIElement<W> + 'b>> {
        let start_len = self.available_segments.len();
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
            (true, 1) => {
                let graph_element = state_segment_edit(self.available_segments.index(0), self.ui_events.clone());
                Some(Box::new(graph_element))
            },
            (_, n) => {
                if n > 0 && self.selected.is_none() && start_len == 0 {
                    self.selected = Some(0);
                }
                self.ui_events.send(UIMessage::UIUpdate);
                None
            }
        }
    }
}

impl<'a, T, L, W> UIElement<W> for ElementSegmentSelector<'a, T, L, W>
    where L: Location,
          T: Segment<L, Item=Position<L>, Output=Position<L>> + IntoGroup<Float32> + 'static,
          W: Write
{
    fn draw(&self, screen: &mut AlternateScreen<W>) {
        let ui_box = &self.ui_box;
        for (i, segment) in self.available_segments.iter().enumerate() {
            let line = ui_box.y + (i as u16);
            write!(screen, "{}", termion::cursor::Goto(ui_box.x, line));
            write!(screen, "{}", color::Fg(color::Blue));
            let selector = match self.selected {
                Some(pos) if pos == i => ">",
                _ => " "
            };
            write!(screen, "{} {}", selector, segment.name());
        }
    }

    fn update<'b>(&'b mut self, message: &UIMessage) -> Option<Box<UIElement<W> + 'b>> {
        match message {
            UIMessage::UIDataSourceUpdate(id) if *id == self.data_source_id => self.data_source_update(),
            UIMessage::UIKeyPress(Key::Char('\n')) => {
                self.selected.map(move |pos| {
                    self.selected = None;
                    Box::new(state_segment_edit(self.available_segments.index(pos), self.ui_events.clone())) as Box<UIElement<W> + 'b>
                })
            },
            UIMessage::UIKeyPress(Key::Char(c)) if *c == 'j' || *c == 'k' => {
                let change = match *c {
                    'j' => 1,
                    _ => -1
                };
                let cap = (self.available_segments.len() - 1) as i32;
                self.selected = self.selected.map(|pos| {
                    self.ui_events.send(UIMessage::UIUpdate);
                    (pos as i32 + change).max(0).min(cap) as usize
                });
                None
            },
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

fn state_segment_edit<'a, L, T>(segment: &'a T, ui_events: Sender<UIMessage>) ->
    BarWindow<T::GroupType, FatalityCutter, &'a T>
    where L: Location,
          T: Segment<L, Item=Position<L>, Output=Position<L>> + IntoGroup<Float32>,
{
    let into_group = |s: &T| s.group_avg(40 as GroupSize);
    let cutter = FatalityCutter::new();
    BarWindow::new(segment, into_group, ui_events.clone(), whole_window(), cutter)
}

fn initial_ui_elements<'a, 'b, L, T, W>(ui_events: Sender<UIMessage>,
                                        segments: Receiver<T>,
                                        loaded_segments: &'a mut Vec<T>) ->
    Vec<Box<UIElement<W> + 'b>>
    where L: Location + 'static,
          T: Segment<L, Item=Position<L>, Output=Position<L>> + IntoGroup<Float32> + 'static,
          W: Write + 'static,
          'a: 'b
{
    vec![Box::new(ElementQuit::new(ui_events.clone())),
         Box::new(ElementSegmentSelector::new(segments, loaded_segments,
                                              ui_events.clone(), whole_window()))]
}

fn whole_window() -> UIBox {
    UIBox{x: 1, y: 1, width: 40, height: 20}
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (tx, rx) = mpsc::channel();
    let mut screen = window_init();
    /*load_test_tracks(tx);*/
    let _loader = thread::spawn(move || xml::segments_from_file(&args[1], tx));
    window_run(rx, &mut screen);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn average() {
        let values = vec![(1.0, Duration::seconds(6)),
                          (2.0, Duration::seconds(1)),
                          (4.0, Duration::seconds(1))];
        assert_eq!(values.into_iter().fold((0.0, Duration::seconds(0)), average_folder),
                   (1.5, Duration::seconds(8)));
    }
}
