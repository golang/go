// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sql_test

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"strings"
	"time"
)

var ctx = context.Background()
var db *sql.DB

func ExampleDB_QueryContext() {
	age := 27
	rows, err := db.QueryContext(ctx, "SELECT name FROM users WHERE age=?", age)
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()
	names := make([]string, 0)
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			log.Fatal(err)
		}
		names = append(names, name)
	}
	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s are %d years old", strings.Join(names, ", "), age)
}

func ExampleDB_QueryRowContext() {
	id := 123
	var username string
	var created time.Time
	err := db.QueryRowContext(ctx, "SELECT username, created_at FROM users WHERE id=?", id).Scan(&username, &created)
	switch {
	case err == sql.ErrNoRows:
		log.Printf("No user with id %d", id)
	case err != nil:
		log.Fatal(err)
	default:
		fmt.Printf("Username is %s, account created on %s\n", username, created)
	}
}

func ExampleDB_ExecContext() {
	id := 47
	result, err := db.ExecContext(ctx, "UPDATE balances SET balance = balance + 10 WHERE user_id = ?", id)
	if err != nil {
		log.Fatal(err)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		log.Fatal(err)
	}
	if rows != 1 {
		panic(err)
	}
}

func ExampleDB_Query_multipleResultSets() {
	age := 27
	q := `
create temp table uid (id bigint); -- Create temp table for queries.
insert into uid
select id from users where age < ?; -- Populate temp table.

-- First result set.
select
	users.id, name
from
	users
	join uid on users.id = uid.id
;

-- Second result set.
select 
	ur.user, ur.role
from
	user_roles as ur
	join uid on uid.id = ur.user
;
	`
	rows, err := db.Query(q, age)
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	for rows.Next() {
		var (
			id   int64
			name string
		)
		if err := rows.Scan(&id, &name); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("id %d name is %s\n", id, name)
	}
	if !rows.NextResultSet() {
		log.Fatal("expected more result sets", rows.Err())
	}
	var roleMap = map[int64]string{
		1: "user",
		2: "admin",
		3: "gopher",
	}
	for rows.Next() {
		var (
			id   int64
			role int64
		)
		if err := rows.Scan(&id, &role); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("id %d has role %s\n", id, roleMap[role])
	}
	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}
}

func ExampleDB_PingContext() {
	ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		log.Fatal(err)
	}
}

func ExampleConn_BeginTx() {
	tx, err := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		log.Fatal(err)
	}
	id := 37
	_, execErr := tx.Exec(`UPDATE users SET status = ? WHERE id = ?`, "paid", id)
	if execErr != nil {
		_ = tx.Rollback()
		log.Fatal(execErr)
	}
	if err := tx.Commit(); err != nil {
		log.Fatal(err)
	}
}

func ExampleConn_ExecContext() {
	// A *DB is a pool of connections. Call Conn to reserve a connection for
	// exclusive use.
	conn, err := db.Conn(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close() // Return the connection to the pool.
	id := 41
	result, err := conn.ExecContext(ctx, `UPDATE balances SET balance = balance + 10 WHERE user_id = ?`, id)
	if err != nil {
		log.Fatal(err)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		log.Fatal(err)
	}
	if rows != 1 {
		panic(err)
	}
}

func ExampleTx_ExecContext() {
	tx, err := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		log.Fatal(err)
	}
	id := 37
	_, execErr := tx.ExecContext(ctx, "UPDATE users SET status = ? WHERE id = ?", "paid", id)
	if execErr != nil {
		if rollbackErr := tx.Rollback(); rollbackErr != nil {
			log.Printf("Could not roll back: %v\n", rollbackErr)
		}
		log.Fatal(execErr)
	}
	if err := tx.Commit(); err != nil {
		log.Fatal(err)
	}
}

func ExampleTx_Rollback() {
	tx, err := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		log.Fatal(err)
	}
	id := 53
	_, err = tx.ExecContext(ctx, "UPDATE drivers SET status = ? WHERE id = ?", "assigned", id)
	if err != nil {
		if rollbackErr := tx.Rollback(); rollbackErr != nil {
			log.Printf("Could not roll back: %v\n", rollbackErr)
		}
		log.Fatal(err)
	}
	_, err = tx.ExecContext(ctx, "UPDATE pickups SET driver_id = $1", id)
	if err != nil {
		if rollbackErr := tx.Rollback(); rollbackErr != nil {
			log.Printf("Could not roll back: %v\n", rollbackErr)
		}
		log.Fatal(err)
	}
	if err := tx.Commit(); err != nil {
		log.Fatal(err)
	}
}

func ExampleStmt() {
	// In normal use, create one Stmt when your process starts.
	stmt, err := db.PrepareContext(ctx, "SELECT username FROM users WHERE id = ?")
	if err != nil {
		log.Fatal(err)
	}
	defer stmt.Close()
	// Then reuse it each time you need to issue the query.
	id := 43
	var username string
	err = stmt.QueryRowContext(ctx, id).Scan(&username)
	switch {
	case err == sql.ErrNoRows:
		log.Printf("No user with that ID.")
	case err != nil:
		log.Fatal(err)
	default:
		fmt.Printf("Username is %s\n", username)
	}
}

func ExampleStmt_QueryRowContext() {
	// In normal use, create one Stmt when your process starts.
	stmt, err := db.PrepareContext(ctx, "SELECT username FROM users WHERE id = ?")
	if err != nil {
		log.Fatal(err)
	}
	// Then reuse it each time you need to issue the query.
	id := 43
	var username string
	err = stmt.QueryRowContext(ctx, id).Scan(&username)
	switch {
	case err == sql.ErrNoRows:
		log.Printf("No user with that ID.")
	case err != nil:
		log.Fatal(err)
	default:
		fmt.Printf("Username is %s\n", username)
	}
}

func ExampleRows() {
	age := 27
	rows, err := db.QueryContext(ctx, "SELECT name FROM users WHERE age=?", age)
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()
	names := make([]string, 0)
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			log.Fatal(err)
		}
		names = append(names, name)
	}
	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s are %d years old", strings.Join(names, ", "), age)
}
