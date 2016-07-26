# Go Heap Viewer Client

This directory contains the client Typescript code for the Go
heap viewer.

## Typescript Tooling

Below are instructions for downloading tooling and files to
help make the development process more convenient. These tools
are not required for contributing or running the heap viewer-
they are just meant as development aids.

## Node and NPM

We use npm to manage the dependencies for these tools. There are
a couple of ways of installing npm on your system, but we recommend
using nvm.

Run the following command to install nvm:

    [shell]$ curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.31.3/install.sh | bash

or see the instructions on [the nvm github page](github.com/creationix/nvm)
for alternative methods. This will put the nvm tool in your home directory
and edit your path to add nvm, node and other tools you install using them.
Once nvm is installed, use

    [shell]$ nvm install node

then

    [shell]$ nvm use node

to install node.js.

Once node is installed, you can install typescript using

    [shell]$ npm install -g typescript

Finally, import type definitions into this project by running

    [shell]$ npm install

in this directory. They will be imported into the node_packages directory
and be automatically available to the Typescript compiler.