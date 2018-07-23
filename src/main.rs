extern crate termion;
extern crate rand;
extern crate core;
extern crate num;

use std::sync::mpsc::{self, Sender, Receiver};
use std::time::SystemTime;

use termion::{color, style};
use termion::screen::*;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use std::string::String;
use rand::distributions::Range;
use core::ops::Index;
use std::ops::{Div, Add, Mul};
use std::iter::FromIterator;
use num::Zero;

use std::io::{Write, stdout, stdin};
use std::convert::From;

type GroupSize = usize;
type Float32 = f32;

struct VecGroup<T> {
    height: T,
    children: Option<Vec<VecGroup<T>>>
}

trait Location {
    fn distance(a: &Self, b: &Self) -> Float32;
}

/* Collection: Index<usize, Output=Group<T>>*/
trait Group<T> {
    type Collection: Index<usize>;
    fn height(self) -> T;
    fn children(self) -> Option<Self::Collection>;
    fn len(self) -> GroupSize;
}

impl<T> Group<T> for VecGroup<T> {
    type Collection = Vec<VecGroup<T>>;
    fn height(self) -> T {
        self.height
    }
    fn children(self) -> Option<Self::Collection> {
        self.children
    }
    fn len(self) -> GroupSize {
        self.children.map_or(0, |v| v.len())
    }
}

trait IntoGroup<T> {
    type GroupType: Group<T>;
    fn group_avg(self, groups: GroupSize) -> Self::GroupType;
}

impl<T> IntoGroup<Float32> for Vec<Position<T>>
    where T: Location {
    type GroupType = VecGroup<Float32>;
    fn group_avg(self, groups: GroupSize) -> Self::GroupType {
        let mut last: Option<Position<T>> = None;
        let mut distances = Vec::new();
        let mut sum = 0;
        for current in self.into_iter() {
            last.map(|prev| distances.push(Location::distance(&prev.location, &current.location)));
            last = Some(current);
        }
        let leaf_group = |height| VecGroup{height: height, children: None};
        let height = distances.iter().fold((0.0, 0 as usize), average_folder).0;
        let children = Vec::from_iter(distances.into_iter().map(leaf_group));
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
        0 => sample.clone(),
        _ => (f32::from(new_count as u16))*(current + (sample / f32::from(count as u16)))
    };
    (avg, new_count)
}

impl Location for u32 {
    fn distance(a: &Self, b: &Self) -> Float32 {
        (b - a) as Float32
    }
}

struct Position<T: Location> {
    time: SystemTime,
    location: T
}

trait Segment {
    fn name(&self) -> String;
}

impl<T: Location> Segment for Vec<Position<T>> {
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

fn show_segment<W: Write, T: Segment>(screen: &mut AlternateScreen<W>, segment: &T) {
    write!(screen, "{}", &segment.name());
    screen.flush().unwrap();
}

fn window_run<T: Segment, W: Write>(segments: Receiver<T>, screen: &mut AlternateScreen<W>) {
    let mut available_segments = Vec::new();
    write!(screen, "{}", termion::clear::All);
    for segment in segments.iter() {
        if available_segments.len() == 1 {
            show_segment(screen, &available_segments[available_segments.len() - 1]);
        }
        available_segments.push(segment);
        if available_segments.len() > 1 {
            show_segment(screen, &available_segments[available_segments.len() - 1]);
        }
    }
    match available_segments.len() {
        0 => (),
        1 => state_segment_edit(screen, &available_segments[0]),
        _ => state_select_segment(screen)
    }
}

fn state_segment_edit<T: Segment, W: Write>(screen: &mut AlternateScreen<W>,
                                             segment: &T) {
    write!(screen, "{}{}", termion::clear::All, termion::cursor::Goto(1,1));
    write!(screen, "segment length: {}", 12);
    let bars = 20;
    /*segment.group_avg(bars);*/
    screen.flush().unwrap();
    state_wait_for_exit(screen);
}

fn state_select_segment<W: Write>(screen: &mut AlternateScreen<W>) {
    state_wait_for_exit(screen);
}

fn state_wait_for_exit<W: Write>(screen: &mut AlternateScreen<W>) {
    for c in stdin().keys() {
        if Key::Char('q') == c.unwrap() {
            break
        }
        screen.flush().unwrap();
    }
}

fn main() {
    let (tx, rx) = mpsc::channel();
    let mut screen = window_init();
    load_test_tracks(tx);
    window_run(rx, &mut screen);
}
