use core::ops::Index;
use std::time::SystemTime;

pub type Float32 = f32;

#[derive(Clone)]
pub struct Position<T: Location> {
    pub time: SystemTime,
    pub location: T
}

pub trait Segment<T: Location>: Index<usize> + IntoIterator + Clone + Send {
    fn name(&self) -> String;
}

pub trait Location: Clone + Send {
    fn distance(a: &Self, b: &Self) -> Float32;
}
