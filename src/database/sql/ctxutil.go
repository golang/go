// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sql

import (
	"context"
	"database/sql/driver"
	"errors"
)

func ctxDriverPrepare(ctx context.Context, ci driver.Conn, query string) (driver.Stmt, error) {
	if ciCtx, is := ci.(driver.ConnPrepareContext); is {
		return ciCtx.PrepareContext(ctx, query)
	}
	if ctx.Done() == context.Background().Done() {
		return ci.Prepare(query)
	}

	type R struct {
		err   error
		panic interface{}
		si    driver.Stmt
	}

	rc := make(chan R, 1)
	go func() {
		r := R{}
		defer func() {
			if v := recover(); v != nil {
				r.panic = v
			}
			rc <- r
		}()
		r.si, r.err = ci.Prepare(query)
	}()
	select {
	case <-ctx.Done():
		go func() {
			<-rc
			close(rc)
		}()
		return nil, ctx.Err()
	case r := <-rc:
		if r.panic != nil {
			panic(r.panic)
		}
		return r.si, r.err
	}
}

func ctxDriverExec(ctx context.Context, execer driver.Execer, query string, nvdargs []driver.NamedValue) (driver.Result, error) {
	if execerCtx, is := execer.(driver.ExecerContext); is {
		return execerCtx.ExecContext(ctx, query, nvdargs)
	}
	dargs, err := namedValueToValue(nvdargs)
	if err != nil {
		return nil, err
	}
	if ctx.Done() == context.Background().Done() {
		return execer.Exec(query, dargs)
	}

	type R struct {
		err   error
		panic interface{}
		resi  driver.Result
	}

	rc := make(chan R, 1)
	go func() {
		r := R{}
		defer func() {
			if v := recover(); v != nil {
				r.panic = v
			}
			rc <- r
		}()
		r.resi, r.err = execer.Exec(query, dargs)
	}()
	select {
	case <-ctx.Done():
		go func() {
			<-rc
			close(rc)
		}()
		return nil, ctx.Err()
	case r := <-rc:
		if r.panic != nil {
			panic(r.panic)
		}
		return r.resi, r.err
	}
}

func ctxDriverQuery(ctx context.Context, queryer driver.Queryer, query string, nvdargs []driver.NamedValue) (driver.Rows, error) {
	if queryerCtx, is := queryer.(driver.QueryerContext); is {
		return queryerCtx.QueryContext(ctx, query, nvdargs)
	}
	dargs, err := namedValueToValue(nvdargs)
	if err != nil {
		return nil, err
	}
	if ctx.Done() == context.Background().Done() {
		return queryer.Query(query, dargs)
	}

	type R struct {
		err   error
		panic interface{}
		rowsi driver.Rows
	}

	rc := make(chan R, 1)
	go func() {
		r := R{}
		defer func() {
			if v := recover(); v != nil {
				r.panic = v
			}
			rc <- r
		}()
		r.rowsi, r.err = queryer.Query(query, dargs)
	}()
	select {
	case <-ctx.Done():
		go func() {
			<-rc
			close(rc)
		}()
		return nil, ctx.Err()
	case r := <-rc:
		if r.panic != nil {
			panic(r.panic)
		}
		return r.rowsi, r.err
	}
}

func ctxDriverStmtExec(ctx context.Context, si driver.Stmt, nvdargs []driver.NamedValue) (driver.Result, error) {
	if siCtx, is := si.(driver.StmtExecContext); is {
		return siCtx.ExecContext(ctx, nvdargs)
	}
	dargs, err := namedValueToValue(nvdargs)
	if err != nil {
		return nil, err
	}
	if ctx.Done() == context.Background().Done() {
		return si.Exec(dargs)
	}

	type R struct {
		err   error
		panic interface{}
		resi  driver.Result
	}

	rc := make(chan R, 1)
	go func() {
		r := R{}
		defer func() {
			if v := recover(); v != nil {
				r.panic = v
			}
			rc <- r
		}()
		r.resi, r.err = si.Exec(dargs)
	}()
	select {
	case <-ctx.Done():
		go func() {
			<-rc
			close(rc)
		}()
		return nil, ctx.Err()
	case r := <-rc:
		if r.panic != nil {
			panic(r.panic)
		}
		return r.resi, r.err
	}
}

func ctxDriverStmtQuery(ctx context.Context, si driver.Stmt, nvdargs []driver.NamedValue) (driver.Rows, error) {
	if siCtx, is := si.(driver.StmtQueryContext); is {
		return siCtx.QueryContext(ctx, nvdargs)
	}
	dargs, err := namedValueToValue(nvdargs)
	if err != nil {
		return nil, err
	}
	if ctx.Done() == context.Background().Done() {
		return si.Query(dargs)
	}

	type R struct {
		err   error
		panic interface{}
		rowsi driver.Rows
	}

	rc := make(chan R, 1)
	go func() {
		r := R{}
		defer func() {
			if v := recover(); v != nil {
				r.panic = v
			}
			rc <- r
		}()
		r.rowsi, r.err = si.Query(dargs)
	}()
	select {
	case <-ctx.Done():
		go func() {
			<-rc
			close(rc)
		}()
		return nil, ctx.Err()
	case r := <-rc:
		if r.panic != nil {
			panic(r.panic)
		}
		return r.rowsi, r.err
	}
}

var errLevelNotSupported = errors.New("sql: selected isolation level is not supported")

func ctxDriverBegin(ctx context.Context, ci driver.Conn) (driver.Tx, error) {
	if ciCtx, is := ci.(driver.ConnBeginContext); is {
		return ciCtx.BeginContext(ctx)
	}
	if ctx.Done() == context.Background().Done() {
		return ci.Begin()
	}

	// TODO(kardianos): check the transaction level in ctx. If set and non-default
	// then return an error here as the BeginContext driver value is not supported.

	type R struct {
		err   error
		panic interface{}
		txi   driver.Tx
	}
	rc := make(chan R, 1)
	go func() {
		r := R{}
		defer func() {
			if v := recover(); v != nil {
				r.panic = v
			}
			rc <- r
		}()
		r.txi, r.err = ci.Begin()
	}()
	select {
	case <-ctx.Done():
		go func() {
			<-rc
			close(rc)
		}()
		return nil, ctx.Err()
	case r := <-rc:
		if r.panic != nil {
			panic(r.panic)
		}
		return r.txi, r.err
	}
}

func namedValueToValue(named []driver.NamedValue) ([]driver.Value, error) {
	dargs := make([]driver.Value, len(named))
	for n, param := range named {
		if len(param.Name) > 0 {
			return nil, errors.New("sql: driver does not support the use of Named Parameters")
		}
		dargs[n] = param.Value
	}
	return dargs, nil
}
