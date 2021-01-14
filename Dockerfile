FROM buildpack-deps:buster

WORKDIR /pysprint-dev

COPY . .

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN set -eux; \
    url="https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init"; \
    wget "$url"; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --default-toolchain nightly; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version;

RUN apt-get update && apt-get install -y \
    python3-pip \
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y

RUN python3 -m pip install -e .[optional] && python -c "import pysprint;pysprint.print_info()"