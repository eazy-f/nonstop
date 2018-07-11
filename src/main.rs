extern crate termion;
extern crate rand;

use std::sync::mpsc::{self, Sender, Receiver};
use std::time::SystemTime;

use termion::{color, style};
use termion::screen::*;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use std::string::String;
use rand::distributions::Range;

use std::io::{Write, stdout, stdin};

trait Location {
    fn distance(a: Self, b: Self) -> f32;
}

impl Location for u32 {
    fn distance(a: u32, b: u32) -> f32 {
        (b - a) as f32
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
    let stdin = stdin();
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
    if available_segments.len() <= 1 {
        write!(screen, "{}{}", termion::clear::All, termion::cursor::Goto(1,1));
        let text = match available_segments.len() {
            0 => "No segments found",
            _ => "Trivial choice"
        };
        write!(screen, "{}", text);
        screen.flush().unwrap();
    }
    for c in stdin.keys() {
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
