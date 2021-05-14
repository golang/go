# Copyright 2021 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

FROM coqorg/coq:8.13.2

RUN git clone https://github.com/mit-plv/fiat-crypto
RUN cd fiat-crypto && git checkout c076f3550bea2bb7f4cb5766a32594b9e67694f2
RUN cd fiat-crypto && git submodule update --init --recursive
RUN cd fiat-crypto && eval $(opam env) && make -j4 standalone-ocaml SKIP_BEDROCK2=1

ENTRYPOINT ["fiat-crypto/src/ExtractionOCaml/unsaturated_solinas"]
