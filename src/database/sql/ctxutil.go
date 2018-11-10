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
	si, err := ci.Prepare(query)
	if err == nil {
		select {
		default:
		case <-ctx.Done():
			si.Close()
			return nil, ctx.Err()
		}
	}
	return si, err
}

func ctxDriverExec(ctx context.Context, execer driver.Execer, query string, nvdargs []driver.NamedValue) (driver.Result, error) {
	if execerCtx, is := execer.(driver.ExecerContext); is {
		return execerCtx.ExecContext(ctx, query, nvdargs)
	}
	dargs, err := namedValueToValue(nvdargs)
	if err != nil {
		return nil, err
	}

	select {
	default:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return execer.Exec(query, dargs)
}

func ctxDriverQuery(ctx context.Context, queryer driver.Queryer, query string, nvdargs []driver.NamedValue) (driver.Rows, error) {
	if queryerCtx, is := queryer.(driver.QueryerContext); is {
		ret, err := queryerCtx.QueryContext(ctx, query, nvdargs)
		return ret, err
	}
	dargs, err := namedValueToValue(nvdargs)
	if err != nil {
		return nil, err
	}

	select {
	default:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return queryer.Query(query, dargs)
}

func ctxDriverStmtExec(ctx context.Context, si driver.Stmt, nvdargs []driver.NamedValue) (driver.Result, error) {
	if siCtx, is := si.(driver.StmtExecContext); is {
		return siCtx.ExecContext(ctx, nvdargs)
	}
	dargs, err := namedValueToValue(nvdargs)
	if err != nil {
		return nil, err
	}

	select {
	default:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return si.Exec(dargs)
}

func ctxDriverStmtQuery(ctx context.Context, si driver.Stmt, nvdargs []driver.NamedValue) (driver.Rows, error) {
	if siCtx, is := si.(driver.StmtQueryContext); is {
		return siCtx.QueryContext(ctx, nvdargs)
	}
	dargs, err := namedValueToValue(nvdargs)
	if err != nil {
		return nil, err
	}

	select {
	default:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return si.Query(dargs)
}

var errLevelNotSupported = errors.New("sql: selected isolation level is not supported")

func ctxDriverBegin(ctx context.Context, opts *TxOptions, ci driver.Conn) (driver.Tx, error) {
	if ciCtx, is := ci.(driver.ConnBeginTx); is {
		dopts := driver.TxOptions{}
		if opts != nil {
			dopts.Isolation = driver.IsolationLevel(opts.Isolation)
			dopts.ReadOnly = opts.ReadOnly
		}
		return ciCtx.BeginTx(ctx, dopts)
	}

	if ctx.Done() == context.Background().Done() {
		return ci.Begin()
	}

	if opts != nil {
		// Check the transaction level. If the transaction level is non-default
		// then return an error here as the BeginTx driver value is not supported.
		if opts.Isolation != LevelDefault {
			return nil, errors.New("sql: driver does not support non-default isolation level")
		}

		// If a read-only transaction is requested return an error as the
		// BeginTx driver value is not supported.
		if opts.ReadOnly {
			return nil, errors.New("sql: driver does not support read-only transactions")
		}
	}

	txi, err := ci.Begin()
	if err == nil {
		select {
		default:
		case <-ctx.Done():
			txi.Rollback()
			return nil, ctx.Err()
		}
	}
	return txi, err
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
