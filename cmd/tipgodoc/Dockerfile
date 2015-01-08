FROM golang

RUN apt-get update && apt-get install --no-install-recommends -y -q build-essential git

# golang sets GOPATH=/go
ADD . /go/src/tipgodoc
RUN go install tipgodoc
ENTRYPOINT ["/go/bin/tipgodoc"]
EXPOSE 8080 # what kubernetes expects
