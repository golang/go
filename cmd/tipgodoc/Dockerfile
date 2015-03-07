FROM golang:1.4.2

RUN apt-get update && apt-get install --no-install-recommends -y -q build-essential git

# golang puts its go install here (weird but true)
ENV GOROOT_BOOTSTRAP /usr/src/go

# golang sets GOPATH=/go
ADD . /go/src/tipgodoc
RUN go install tipgodoc
ENTRYPOINT ["/go/bin/tipgodoc"]
# Kubernetes expects us to listen on port 8080
EXPOSE 8080 
