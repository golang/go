#!/usr/bin/env zsh

set -eu -o pipefail

# This is a large series of sed commands to cleanup after successful use of the
# `rf inject` command.  This script will be used to refactor the codebase to
# eliminate global state within the module loader.  Once that effort is
# complete, this script will be removed.

find . -name '*.go' -exec \
  sed -i '
    #
    # CompileAction does not use loaderstate.
    #
    s/CompileAction(loaderstate[^ ]* \*modload.State, /CompileAction(/g
    s/CompileAction(modload.LoaderState[^,]*, /CompileAction(/g
    s/CompileAction(loaderstate[^,]*, /CompileAction(/g
    #
    # cgoAction does not use loaderstate.
    #
    s/cgoAction(loaderstate \*modload\.State, /cgoAction(/g
    s/cgoAction(loaderstate, /cgoAction(/g
    s/cgoAction(loaderstate_, /cgoAction(/g
    #
    # Remove redundant mentions of LoaderState from function call sites.
    #
    s/(modload\.LoaderState_*, loaderstate,/(loaderstate,/g
    s/(modload\.LoaderState_*, moduleLoaderState,/(moduleLoaderState,/g
    s/(modload\.LoaderState_*, modload\.LoaderState/(modload.LoaderState/g
    s/(modload\.LoaderState_*, loaderstate,/(loaderstate,/g
    s/(modload\.LoaderState_*, moduleLoaderState,/(moduleLoaderState,/g
    s/(modload\.LoaderState_*, modload\.LoaderState,/(modload.LoaderState,/g
    s/(loaderstate_* \*modload.State, loaderstate \*modload.State/(loaderstate *modload.State/g
    s/(loaderstate_* \*State, loaderstate \*State/(loaderstate *State/g
    s/(loaderstate_*, loaderstate,/(loaderstate,/g
    s/(LoaderState_*, loaderstate,/(loaderstate,/g
    s/(LoaderState_*, loaderState,/(loaderState,/g
    s/(LoaderState_*, LoaderState,/(LoaderState,/g
    s/(LoaderState_*, LoaderState,/(LoaderState,/g
    s/(moduleLoaderState_*, loaderstate,/(loaderstate,/g
    s/(moduleLoaderState_*, moduleLoaderState,/(moduleLoaderState,/g
  ' {} \;

