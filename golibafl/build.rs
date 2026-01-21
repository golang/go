use anyhow::{anyhow, Result};
use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::{exit, Command};

fn run<S, I, A>(cmd: S, args: I, dir: Option<&String>) -> Result<()>
where
    S: AsRef<OsStr>,
    I: IntoIterator<Item = A>,
    A: AsRef<OsStr>,
{
    let status = if let Some(dir) = dir {
        Command::new(&cmd).args(args).current_dir(dir).status()?
    } else {
        Command::new(&cmd).args(args).status()?
    };

    if status.success() {
        return Ok(());
    }

    let exit_code = status.code().unwrap_or(2);
    Err(anyhow!(
        "command '{}' failed with exit code {}",
        cmd.as_ref().to_string_lossy(),
        exit_code
    ))
}

fn main() -> Result<()> {
    // The harness is produced by `go build -buildmode=c-archive`.
    // On macOS, the Go stdlib (e.g., crypto/x509) may rely on system frameworks.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Security");
    }

    if let Ok(harness_lib) = env::var("HARNESS_LIB") {
        let harness_lib = PathBuf::from(harness_lib);
        let dir = harness_lib
            .parent()
            .ok_or_else(|| anyhow!("HARNESS_LIB must point to a file"))?;

        println!("cargo:rerun-if-env-changed=HARNESS_LIB");
        println!("cargo:rerun-if-env-changed=HARNESS_LINK_SEARCH");
        println!("cargo:rerun-if-env-changed=HARNESS_LINK_LIBS");
        println!("cargo:rerun-if-changed={}", harness_lib.display());
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static=harness");

        if let Ok(extra_search) = env::var("HARNESS_LINK_SEARCH") {
            for dir in extra_search
                .split(':')
                .map(str::trim)
                .filter(|d| !d.is_empty())
            {
                println!("cargo:rustc-link-search=native={dir}");
            }
        }

        if let Ok(extra_libs) = env::var("HARNESS_LINK_LIBS") {
            for lib in extra_libs
                .split(',')
                .map(str::trim)
                .filter(|l| !l.is_empty())
            {
                // Accept values like: "static=stylus" or "dylib=dl"
                println!("cargo:rustc-link-lib={lib}");
            }
        }
        return Ok(());
    }

    const HARNESS_WRAPPER: &str = "harness_fuzz.go";
    // Enable cgo
    env::set_var("CGO_ENABLED", "1");

    let harness_path = env::var("HARNESS").unwrap();

    //rerun_if_changed_recursive(&Path::new(harness_path.as_str()));
    //println!("cargo::rerun-if-changed={}", harness_path);

    // Define the output directory for the Go library
    let out_dir = match env::var("OUT_DIR") {
        Ok(out) => PathBuf::from(out),
        Err(err) => {
            eprintln!("Failed to get OUT_DIR: {err}");
            exit(1);
        }
    };

    // copy our harness wrapper to the harness directory
    run(
        "cp",
        [
            &format!(
                "{}/harness_wrappers/{}",
                env!("CARGO_MANIFEST_DIR"),
                HARNESS_WRAPPER,
            ),
            &harness_path,
        ],
        None,
    )?;

    // Enable usage of different go versions
    let go_binary = env::var("GO_PATH").unwrap_or("go".to_string());

    // Build the Go code as a static library
    let res = run(
        &go_binary,
        [
            "build",
            "-buildmode=c-archive",
            "-tags=libfuzzer,gofuzz",
            "-gcflags=all=-d=libfuzzer", // Enable coverage instrumentation for libfuzzer
            // avoid instrumenting unnecessary packages
            "-gcflags=runtime/cgo=-d=libfuzzer=0",
            "-gcflags=runtime/pprof=-d=libfuzzer=0",
            "-gcflags=runtime/race=-d=libfuzzer=0",
            "-gcflags=syscall=-d=libfuzzer=0",
            "-o",
            out_dir.join("libharness.a").as_os_str().to_str().unwrap(),
        ],
        Some(&harness_path),
    );

    // cleanup: remove our wrapper
    run(
        "rm",
        ["-f", &format!("{}/{}", harness_path, HARNESS_WRAPPER)],
        None,
    )?;

    // display error after we cleaned up the files we created
    res?;

    // Tell cargo to look for the library in the output directory
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // Tell cargo to link the static Go library
    println!("cargo:rustc-link-lib=static=harness");

    Ok(())
}
