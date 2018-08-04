use std::sync::mpsc::Sender;
use std::path::Path;
use std::time::{SystemTime, Duration};
use std::str::FromStr;
use std::borrow::Cow;

use std::thread;
use std::time::Duration as D;

use quick_xml::Reader;
use quick_xml::events::Event;

use chrono::prelude::*;

use segment::{Location, Position, Float32};

#[derive(Clone)]
pub struct GpxLocation {
    lat: f32,
    lon: f32
}

impl Location for GpxLocation {
    fn distance(a: &Self, b: &Self) -> Float32 {
        let lat_diff = b.lat - a.lat;
        let lon_diff = b.lon - a.lon;
        (lat_diff*lat_diff + lon_diff*lon_diff).sqrt()
    }
}

pub type GpxPosition = Position<GpxLocation>;

fn float_from_xml_text(value: Cow<[u8]>) -> f32 {
    f32::from_str(&String::from_utf8_lossy(&value)).unwrap()
}

pub fn segments_from_file(filename: &String, segments: Sender<Vec<GpxPosition>>)
{
    #[derive(Clone)]
    enum Param {Name, Elevation, Time};
    #[derive(Clone)]
    enum State {Document, Gpx, Track, Segment, Point, PointParam(Param)};
    let mut reader = Reader::from_file(Path::new(filename)).unwrap();
    let mut buffer = Vec::new();
    let mut state = State::Document;
    let mut location = None;
    let mut time = None;
    let mut segment = Vec::new();
    loop {
        match (reader.read_event(&mut buffer), state.clone()) {
            (Ok(Event::Start(ref e)), State::Document) if e.name() == b"gpx" => {
                state = State::Gpx;
            },
            (Ok(Event::End(ref e)), State::Gpx) if e.name() == b"gpx" => {
                state = State::Document;
            },
            (Ok(Event::Start(ref e)), State::Gpx) if e.name() == b"trk" => {
                state = State::Track;
            },
            (Ok(Event::End(ref e)), State::Track) if e.name() == b"trk" => {
                state = State::Gpx;
            },
            (Ok(Event::Start(ref e)), State::Track) if e.name() == b"trkseg" => {
                state = State::Segment;
            },
            (Ok(Event::End(ref e)), State::Segment) if e.name() == b"trkseg" => {
                segments.send(segment);
                segment = Vec::new();
                state = State::Track;
            },
            (Ok(Event::Start(ref e)), State::Segment) if e.name() == b"trkpt" => {
                let mut lat = None;
                let mut lon = None;
                e.attributes().for_each(|attr| {
                    let a = attr.unwrap();
                    match a.key {
                        b"lat" =>
                            lat = Some(float_from_xml_text(a.value)),
                        b"lon" =>
                            lon = Some(float_from_xml_text(a.value)),
                        _ => ()
                    }
                });
                if lat.is_some() && lon.is_some() {
                    location = Some(GpxLocation{lat: lat.unwrap(), lon: lon.unwrap()});
                }
                state = State::Point;
            },
            (Ok(Event::End(ref e)), State::Point) if e.name() == b"trkpt" => {
                location.map(|loc| {
                    time.map(|time| {
                        segment.push(GpxPosition{location: loc, time: time});
                    });
                });
                location = None;
                time = None;
                state = State::Segment;
            },
            (Ok(Event::Start(ref e)), State::Point) if e.name() == b"name" => {
                state = State::PointParam(Param::Name);
            },
            (Ok(Event::Start(ref e)), State::Point) if e.name() == b"time" => {
                state = State::PointParam(Param::Time);
            },
            (Ok(Event::End(ref e)), State::PointParam(_)) => {
                state = State::Point;
            },
            (Ok(Event::Text(e)), State::PointParam(Param::Time)) => {
                let text = e.unescape_and_decode(&reader).unwrap();
                time = Some(text.parse::<DateTime<Utc>>().unwrap_or(Utc::now()));
            },
            (Ok(Event::Eof), _) => break,
            _ => ()
        }
    }
}
