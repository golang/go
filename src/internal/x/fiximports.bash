#!/bin/bash

# To fix import paths when importing new snapshots from the golang.org/x
# repositories, run this script in the current directory.

sed -i 's,"golang\.org/x,"internal/x,g' $(grep -lr 'golang.org')
