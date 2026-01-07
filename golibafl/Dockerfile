FROM ubuntu:24.10


# Install dependencies for Go and Rust
RUN apt-get update
RUN apt-get install -y\
    bash \
    curl \
    git \
    gcc \
    golang

# Set the working directory
WORKDIR /golibafl

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the source code into the container
COPY . .

ARG HARNESS=harnesses/prometheus
ENV HARNESS=$HARNESS

# Compile golibafl and the harness
RUN cargo build --release

# Fuzz!
CMD ["/golibafl/target/release/golibafl", "fuzz"]
