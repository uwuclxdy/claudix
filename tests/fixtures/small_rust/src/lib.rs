pub mod math;

pub fn greet(name: &str) -> String {
    format!("hello {name}")
}

pub fn farewell(name: &str) -> String {
    format!("goodbye {name}")
}
