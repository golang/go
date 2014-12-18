# Dockerfile extending the generic Go image with application files for a
# single application.
FROM google/appengine-go

RUN apt-get update
RUN apt-get install --no-install-recommends -y -q curl build-essential git
RUN mkdir /goroot && curl https://storage.googleapis.com/golang/go1.2.2.linux-amd64.tar.gz | tar xvzf - -C /goroot --strip-components=1
RUN mkdir /gopath

ENV GOROOT /goroot
ENV GOPATH /gopath
ENV PATH $PATH:$GOROOT/bin:$GOPATH/bin

WORKDIR /app
ADD . /app
RUN /bin/bash /app/_ah/build.sh
