# GNU Make syntax:
nullstring :=
space := $(nullstring) # a space at the end
QUOTED_GOROOT=$(subst $(space),\ ,$(GOROOT))

all: build run

build:
	bash $(QUOTED_GOROOT)/src/make.bash

run:
	bash $(QUOTED_GOROOT)/src/run.bash

clean:
	bash $(QUOTED_GOROOT)/src/clean.bash

