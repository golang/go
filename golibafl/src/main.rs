use clap::{Parser, Subcommand};
use libafl::{
    corpus::{CachedOnDiskCorpus, Corpus, OnDiskCorpus},
    executors::{inprocess::InProcessExecutor, ExitKind, ShadowExecutor},
    events::{
        Event, EventFirer, EventWithStats, LlmpRestartingEventManager, ProgressReporter,
        SendExiting,
    },
    feedback_or_fast,
    feedbacks::{CrashFeedback, MaxMapFeedback},
    fuzzer::{Fuzzer, StdFuzzer},
    inputs::{BytesInput, HasTargetBytes},
    mutators::scheduled::HavocScheduledMutator,
    nonzero,
    prelude::{
        havoc_mutations, powersched::PowerSchedule, tokens_mutations, CalibrationStage, CanTrack,
        ClientDescription, EventConfig, I2SRandReplace, IndexesLenTimeMinimizerScheduler, Launcher,
        RandBytesGenerator, SimpleMonitor, StdMOptMutator, StdMapObserver, StdWeightedScheduler,
        TimeFeedback, TimeObserver, Tokens,
    },
    stages::{mutational::StdMutationalStage, ShadowTracingStage, StdPowerMutationalStage},
    state::{HasCorpus, HasExecutions, HasSolutions, Stoppable, StdState},
    Error, HasMetadata,
};
use libafl_bolts::{
    prelude::{Cores, StdShMemProvider},
    rands::StdRand,
    shmem::ShMemProvider,
    tuples::{tuple_list, Merge},
};
use libafl_targets::{
    autotokens, extra_counters, libfuzzer::libfuzzer_test_one_input, libfuzzer_initialize,
    CmpLogObserver, COUNTERS_MAPS,
};
use mimalloc::MiMalloc;
use std::{
    env, fs,
    fs::read_dir,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use std::{panic, process::Command};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;



// Command line arguments with clap
#[derive(Subcommand, Debug, Clone)]
enum Mode {
    Run {
        #[clap(short, long, value_name = "DIR", default_value = "./input")]
        input: PathBuf,
    },
    Fuzz {
        #[clap(
            short = 'j',
            long,
            value_parser = Cores::from_cmdline,
            help = "Spawn clients in each of the provided cores. Broker runs in the 0th core. 'all' to select all available cores. 'none' to run a client without binding to any core. eg: '1,2-4,6' selects the cores 1,2,3,4,6.",
            name = "CORES",
            default_value = "all",
            )]
        cores: Cores,

        #[clap(
            short = 'p',
            long,
            help = "Choose the broker TCP port, default is 1337",
            name = "PORT",
            default_value = "1337"
        )]
        broker_port: u16,

        #[clap(
            short,
            long,
            value_name = "DIR",
            default_value = "./input",
            help = "Initial corpus directory (will only be read)"
        )]
        input: PathBuf,

        #[clap(
            short,
            long,
            value_name = "OUTPUT",
            default_value = "./output",
            help = "Fuzzer's output directory"
        )]
        output: PathBuf,
    },
    Cov {
        #[clap(short, long, value_name = "OUTPUT", help = "Fuzzer's output directory")]
        output: PathBuf,
        #[clap(
            short,
            long,
            value_name = "HARNESS",
            help = "Fuzzer's harness directory"
        )]
        fuzzer_harness: PathBuf,
        #[clap(
            short,
            long,
            value_name = "COV_PACKAGE",
            help = "Package name the coverage should be filtered for"
        )]
        coverage_filter: Option<String>,
    },
}
// Clap top level struct for args
// `Parser` is needed for the top-level command-line interface
#[derive(Parser, Debug, Clone)]
struct Cli {
    #[command(subcommand)]
    mode: Mode,
}

// Run the corpus without fuzzing
fn run(input: PathBuf) {
    let files = if input.is_dir() {
        input
            .read_dir()
            .expect("Unable to read dir")
            .filter_map(core::result::Result::ok)
            .map(|e| e.path())
            .collect()
    } else {
        vec![input]
    };

    // Call LLVMFuzzerInitialize() if present.
    let args: Vec<String> = env::args().collect();
    if unsafe { libfuzzer_initialize(&args) } == -1 {
        println!("Warning: LLVMFuzzerInitialize failed with -1");
    }

    for f in &files {
        println!("\x1b[33mRunning: {}\x1b[0m", f.display());
        let inp =
            std::fs::read(f).unwrap_or_else(|_| panic!("Unable to read file {}", &f.display()));
        if inp.len() > 1 {
            unsafe {
                libfuzzer_test_one_input(&inp);
            }
        }
    }
}

// Fuzzing function, wrapping the exported libfuzzer functions from golang
#[allow(clippy::too_many_lines)]
#[allow(static_mut_refs)]
fn fuzz(cores: &Cores, broker_port: u16, input: &PathBuf, output: &Path) {
    let args: Vec<String> = env::args().collect();
    if cores.ids.len() == 1 {
        env::set_var("LIBAFL_DEBUG_OUTPUT", "1");
    }
    let crashes_dir = output.join("crashes");
    let count_crash_inputs = |dir: &Path| -> usize {
        fs::read_dir(dir)
            .ok()
            .map(|rd| {
                rd.filter_map(Result::ok)
                    .filter(|e| {
                        !e.file_name()
                            .to_string_lossy()
                            .starts_with('.')
                            && e.file_type().is_ok_and(|t| t.is_file())
                    })
                    .count()
            })
            .unwrap_or(0)
    };
    let computed_initial_crash_inputs = count_crash_inputs(&crashes_dir);
    // The fuzzer process may be respawned by LibAFL's restarting manager. Propagate the initial
    // crash count across respawns so "stop on first crash" stays correct after a restart.
    let initial_crash_inputs = match env::var("GOLIBAFL_INITIAL_CRASH_INPUTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(v) => v,
        None => {
            env::set_var(
                "GOLIBAFL_INITIAL_CRASH_INPUTS",
                computed_initial_crash_inputs.to_string(),
            );
            computed_initial_crash_inputs
        }
    };
    let is_launcher_client = env::var_os("AFL_LAUNCHER_CLIENT").is_some();
    if initial_crash_inputs > 0 && !is_launcher_client {
        eprintln!(
            "Found {} crashing input(s) from a previous run. Saved to {}",
            initial_crash_inputs,
            crashes_dir.display()
        );
        std::process::exit(1);
    }
    // On macOS, LibAFL's `StdShMemProvider` uses a on-disk unix socket at
    // `./libafl_unix_shmem_server`. If a previous run crashed, a stale socket
    // may be left behind and prevent the shmem service from starting.
    //
    // Only attempt to remove it if this process is about to start the shmem
    // service. Child processes (when `.fork(false)` is used) inherit the
    // `AFL_SHMEM_SERVICE_STARTED` env var and must not remove the broker's
    // socket.
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileTypeExt;
        if env::var("AFL_SHMEM_SERVICE_STARTED").is_err() {
            if let Ok(meta) = fs::metadata("libafl_unix_shmem_server") {
                if meta.file_type().is_socket() {
                    let _ = fs::remove_file("libafl_unix_shmem_server");
                }
            }
        }
    }
    let shmem_provider =
        StdShMemProvider::new().unwrap_or_else(|err| panic!("Failed to init shared memory: {err:?}"));
    let monitor = SimpleMonitor::new(|s| println!("{s}"));

    let mut run_client =
        |state: Option<_>,
         mut restarting_mgr: LlmpRestartingEventManager<_, BytesInput, _, _, _>,
         client_description: ClientDescription| {
            // In-process crashes abort the fuzzing instance, and the restarting manager respawns it.
            // Implement `go test -fuzz` semantics: stop the whole run on the first crash.
            if count_crash_inputs(&crashes_dir) > initial_crash_inputs {
                restarting_mgr.on_shutdown()?;
                return Err(Error::shutting_down());
            }

            // trigger Go runtime initialization, which calls __sanitizer_cov_8bit_counters_init to initialize COUNTERS_MAPS
            if unsafe { libfuzzer_initialize(&args) } == -1 {
                println!("Warning: LLVMFuzzerInitialize failed with -1");
            }
            // We assume COUNTERS_MAP len == 1  so that we can use StdMapObserver instead of Multimapobserver to improve performance.
            let counters_map_len = unsafe { COUNTERS_MAPS.len() };
            assert!(
                (counters_map_len == 1),
                "{}",
                format!("Unexpected COUNTERS_MAPS length: {counters_map_len}")
            );
            let edges = unsafe { extra_counters() };
            let edges_observer =
                StdMapObserver::from_mut_slice("edges", edges.into_iter().next().unwrap())
                    .track_indices();

            // Observers
            let time_observer = TimeObserver::new("time");
            let cmplog_observer = CmpLogObserver::new("cmplog", true);
            let map_feedback = MaxMapFeedback::new(&edges_observer);
            let calibration = CalibrationStage::new(&map_feedback);

            let mut feedback = feedback_or_fast!(
                // New maximization map feedback linked to the edges observer and the feedback state
                map_feedback,
                // Time feedback, this one does not need a feedback state
                TimeFeedback::new(&time_observer)
            );

            // A feedback to choose if an input is a solution or not
            let mut objective = feedback_or_fast!(CrashFeedback::new());

            // create a State from scratch
            let mut state = state.unwrap_or_else(|| {
                StdState::new(
                    StdRand::new(),
                    // Corpus that will be evolved
                    CachedOnDiskCorpus::new(
                        format!("{}/queue/{}", output.display(), client_description.id()),
                        4096,
                    )
                    .unwrap(),
                    // Corpus in which we store solutions
                    OnDiskCorpus::new(format!("{}/crashes", output.display())).unwrap(),
                    &mut feedback,
                    &mut objective,
                )
                .unwrap()
            });
            let initial_solutions = state.solutions().count();

            // Setup a randomic Input2State stage
            let i2s = StdMutationalStage::new(HavocScheduledMutator::new(tuple_list!(
                I2SRandReplace::new()
            )));

            // Setup a MOPT mutator
            let mutator = StdMOptMutator::new(
                &mut state,
                havoc_mutations().merge(tokens_mutations()),
                7,
                5,
            )?;

            let power: StdPowerMutationalStage<_, _, BytesInput, _, _, _> =
                StdPowerMutationalStage::new(mutator);

            let scheduler = IndexesLenTimeMinimizerScheduler::new(
                &edges_observer,
                StdWeightedScheduler::with_schedule(
                    &mut state,
                    &edges_observer,
                    Some(PowerSchedule::fast()),
                ),
            );

            // A fuzzer with feedbacks and a corpus scheduler
            let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

            // The closure that we want to fuzz
            let mut harness = |input: &BytesInput| {
                let target = input.target_bytes();
                unsafe {
                    libfuzzer_test_one_input(&target);
                }
                ExitKind::Ok
            };

            let executor = InProcessExecutor::with_timeout(
                &mut harness,
                tuple_list!(edges_observer, time_observer),
                &mut fuzzer,
                &mut state,
                &mut restarting_mgr,
                Duration::new(1, 0),
            )?;

            let mut executor = ShadowExecutor::new(executor, tuple_list!(cmplog_observer));

            // Setup a tracing stage in which we log comparisons
            let tracing = ShadowTracingStage::new();

            let mut stages = tuple_list!(calibration, tracing, i2s, power);

            if state.metadata_map().get::<Tokens>().is_none() {
                let mut toks = Tokens::default();
                toks += autotokens()?;

                if !toks.is_empty() {
                    state.add_metadata(toks);
                }
            }

            // Load corpus from input folder
            // In case the corpus is empty (on first run), reset
            if state.must_load_initial_inputs() {
                let input_is_empty = match read_dir(input) {
                    Ok(mut entries) => entries.next().is_none(),
                    Err(_) => true,
                };
                if input_is_empty {
                    // Generator of printable bytearrays of max size 32
                    let mut generator = RandBytesGenerator::new(nonzero!(32));

                    // Generate 8 initial inputs
                    state
                        .generate_initial_inputs(
                            &mut fuzzer,
                            &mut executor,
                            &mut generator,
                            &mut restarting_mgr,
                            8,
                        )
                        .expect("Failed to generate the initial corpus");
                    println!(
                        "We imported {} inputs from the generator.",
                        state.corpus().count()
                    );
                } else {
                    println!("Loading from {input:?}");
                    // Load from disk
                    state
                        .load_initial_inputs(
                            &mut fuzzer,
                            &mut executor,
                            &mut restarting_mgr,
                            &[input.to_path_buf()],
                        )
                        .unwrap_or_else(|err| {
                            panic!("Failed to load initial corpus at {input:?}: {err:?}");
                        });
                    let disk_inputs = state.corpus().count();
                    println!("We imported {} inputs from disk.", disk_inputs);
                    if disk_inputs == 0 {
                        // Generator of printable bytearrays of max size 32
                        let mut generator = RandBytesGenerator::new(nonzero!(32));

                        // Generate 8 initial inputs
                        state
                            .generate_initial_inputs(
                                &mut fuzzer,
                                &mut executor,
                                &mut generator,
                                &mut restarting_mgr,
                                8,
                            )
                            .expect("Failed to generate the initial corpus");
                        println!(
                            "We imported {} inputs from the generator.",
                            state.corpus().count()
                        );
                    }
                }
            }

            if state.solutions().count() > initial_solutions {
                let executions = *state.executions();
                restarting_mgr.fire(
                    &mut state,
                    EventWithStats::with_current_time(Event::<BytesInput>::Stop, executions),
                )?;
                state.request_stop();
                restarting_mgr.on_shutdown()?;
                return Err(Error::shutting_down());
            }

            let monitor_timeout = Duration::from_secs(15);
            loop {
                if let Err(err) = restarting_mgr.maybe_report_progress(&mut state, monitor_timeout)
                {
                    if matches!(err, Error::ShuttingDown) {
                        restarting_mgr.on_shutdown()?;
                    }
                    return Err(err);
                }

                if let Err(err) =
                    fuzzer.fuzz_one(&mut stages, &mut executor, &mut state, &mut restarting_mgr)
                {
                    if matches!(err, Error::ShuttingDown) {
                        restarting_mgr.on_shutdown()?;
                    }
                    return Err(err);
                }

                if state.solutions().count() > initial_solutions {
                    let executions = *state.executions();
                    restarting_mgr.fire(
                        &mut state,
                        EventWithStats::with_current_time(Event::<BytesInput>::Stop, executions),
                    )?;
                    state.request_stop();
                    restarting_mgr.on_shutdown()?;
                    return Err(Error::shutting_down());
                }
            }
        };
    let launch_res = Launcher::builder()
        .shmem_provider(shmem_provider)
        .configuration(EventConfig::from_name("default"))
        .monitor(monitor)
        .run_client(&mut run_client)
        .cores(cores)
        .broker_port(broker_port)
        .fork(false)
        .build()
        .launch::<BytesInput, _>();

    match &launch_res {
        Ok(()) | Err(Error::ShuttingDown) => (),
        Err(err) => panic!("Failed to run launcher: {err:?}"),
    };

    let crash_inputs = count_crash_inputs(&crashes_dir);

    if crash_inputs > initial_crash_inputs {
        if !is_launcher_client {
            eprintln!(
                "Found {} crashing input(s). Saved to {}",
                crash_inputs - initial_crash_inputs,
                crashes_dir.display()
            );
            std::process::exit(1);
        }
        return;
    }

    if matches!(launch_res, Err(Error::ShuttingDown)) {
        println!("Fuzzing stopped by user. Good bye.");
    }
}

fn cov(output_dir: &Path, harness_dir: &Path, coverage_filter: Option<String>) {
    let mut test_code = String::from(include_str!("../harness_wrappers/harness_test.go"));

    let output_dir = if output_dir.is_relative() {
        &format!(
            "{}/{}",
            env!("CARGO_MANIFEST_DIR"),
            output_dir.as_os_str().to_str().unwrap()
        )
    } else {
        output_dir.as_os_str().to_str().unwrap()
    };

    let harness_dir = harness_dir
        .as_os_str()
        .to_str()
        .expect("Harness dir not valid unicode");

    test_code = test_code.replace("REPLACE_ME", output_dir);

    fs::write(format!("{}/harness_test.go", harness_dir), test_code)
        .expect("Failed to write coverage go file");

    let output = Command::new("go")
        .args(["list", "-deps", "-test"])
        .current_dir(harness_dir)
        .output()
        .expect("Failed to execute go");

    let filter_terms: Vec<&str> = if let Some(coverage_filter) = coverage_filter.as_ref() {
        coverage_filter
            .split(",")
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        Vec::new()
    };

    let deps_raw = String::from_utf8_lossy(&output.stdout);
    let mut packages: Vec<&str> = deps_raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    if !filter_terms.is_empty() {
        packages.retain(|p| filter_terms.iter().any(|t| p.contains(t)));
    }

    let status = Command::new("go")
        .args([
            "test",
            "-tags=gocov",
            "-run=FuzzMe",
            "-cover",
            &format!("-coverpkg={}", packages.join(",")),
            "-coverprofile",
            "cover.out",
        ])
        .current_dir(harness_dir)
        .stdout(Stdio::null())
        .status();

    fs::remove_file(format!("{}/harness_test.go", harness_dir))
        .expect("Failed to remove coverage file");

    // make sure we unpack status after we removed file
    status.expect("Failed to execute go");

    Command::new("go")
        .args(["tool", "cover", "-html", "cover.out", "-o", "cover.html"])
        .current_dir(harness_dir)
        .status()
        .expect("Failed to execute go");

    println!("Coverage files succesfully created in {}", harness_dir)
}

// Entry point wrapping clap and calling fuzz, run or cov
pub fn main() {
    let cli = Cli::parse();

    match cli.mode {
        Mode::Fuzz {
            cores,
            broker_port,
            input,
            output,
        } => fuzz(&cores, broker_port, &input, &output),
        Mode::Run { input } => {
            run(input);
        }
        Mode::Cov {
            output,
            fuzzer_harness,
            coverage_filter,
        } => cov(&output, &fuzzer_harness, coverage_filter),
    }
}
