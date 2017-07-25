FROM golang:latest

ENV SHELL /bin/bash
ENV HOME /root
WORKDIR $HOME

COPY . /go/src/golang.org/x/tools/cmd/getgo

RUN ( \
		cd /go/src/golang.org/x/tools/cmd/getgo \
		&& go build \
		&& mv getgo /usr/local/bin/getgo \
	)

# undo the adding of GOPATH to env for testing
ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV GOPATH ""

# delete /go and /usr/local/go for testing
RUN rm -rf /go /usr/local/go
