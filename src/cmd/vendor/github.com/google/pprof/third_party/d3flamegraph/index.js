// This file exports a stripped-down API surface of d3 and d3-flame-graph,
// using only the functions used by pprof.

export {
  select,
} from "d3-selection";

export {
  default as flamegraph
// If we export from "d3-flame-graph" that exports the "dist" version which
// includes another copy of d3-selection. To avoid including d3-selection
// twice in the output, instead import the "src" version.
} from "d3-flame-graph/src/flamegraph";
