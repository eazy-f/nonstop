use core::ops::Index;
use std::time::SystemTime;

use chrono::prelude::*;

pub type Float32 = f32;

#[derive(Clone)]
pub struct Position<T: Location> {
    pub time: DateTime<Utc>,
    pub location: T
}

pub trait Segment<T: Location>: Index<usize> + IntoIterator + Clone + Send
/*    where for<'a> &'a Segment<T, Output=Self::Output, Item=Self::Item, IntoIter=Self::IntoIter>: IntoIterator */{
    fn name(&self) -> String;
}

pub trait Location: Clone + Send {
    fn distance(a: &Self, b: &Self) -> Float32;
}
