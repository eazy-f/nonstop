extern crate termion;
extern crate rand;
extern crate core;
extern crate num;

use std::sync::mpsc::{self, Sender, Receiver};
use std::time::{SystemTime, Duration};

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
use std::thread::sleep;
use std::slice::Iter;

use std::io::{Write, stdout, stdin};
use std::convert::From;
use std::vec::IntoIter;
use std::fmt::Display;
use std::fmt;
use std::cmp::Ordering;

type GroupSize = usize;
type Float32 = f32;

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

trait Location: Clone {
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
          S: Segment<T, Item=Position<T>, Output=Position<T>> {
    type GroupType = VecGroup<Float32>;
    fn group_avg(self, groups: GroupSize) -> Self::GroupType {
        let mut last: Option<Position<T>> = None;
        let mut distances = Vec::new();
        let mut sum = 0;
        for current in self.into_iter() {
            for prev in last.iter() {
                distances.push(Location::distance(&prev.location, &current.location));
            }
            last = Some(current);
        }
        let leaf_group = |height| VecGroup{height: height, children: None};
        let height = distances[0..groups].iter().fold((0.0, 0 as usize), average_folder).0;
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

trait Segment<T: Location>: Index<usize> + IntoIterator + Clone {
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
    where L: Location, T: Segment<L, Item=Position<L>, Output=Position<L>>, W: Write {
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
        1 => state_segment_edit(screen, available_segments[0].clone()),
        _ => state_select_segment(screen)
    }
}

fn state_segment_edit<T, W, L>(screen: &mut AlternateScreen<W>, segment: T)
    where L: Location, T: Segment<L, Item=Position<L>, Output=Position<L>>, W: Write {
    write!(screen, "{}{}", termion::clear::All, termion::cursor::Goto(1,1));
    write!(screen, "segment length: {}", 12);
    let bars = 20;
    let group = segment.group_avg(bars);
    let max = group.iter().max().unwrap().height;
    let normalized = group.iter().map(|x| (x.height * 100.0) / max);
    for bar in normalized {
        write!(screen, "|{}|", bar);
    }
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
