use claudix::config::Config;

pub fn stub_config() -> Config {
    stub_config_with_model("stub-v1")
}

pub fn stub_config_with_model(model: impl Into<String>) -> Config {
    let mut config = Config::default();
    config.embedding.model = model.into();
    config.embedding.dimensions = 8;
    config
}
