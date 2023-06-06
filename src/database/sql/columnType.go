package sql

import (
	"database/sql/driver"
	"reflect"
)

// ColumnType contains the name and type of a column.
type ColumnType struct {
	name string

	hasNullable       bool
	hasLength         bool
	hasPrecisionScale bool

	nullable     bool
	length       int64
	databaseType string
	precision    int64
	scale        int64
	scanType     reflect.Type
}

// Name returns the name or alias of the column.
func (ci *ColumnType) Name() string {
	return ci.name
}

// Length returns the column type length for variable length column types such
// as text and binary field types. If the type length is unbounded the value will
// be math.MaxInt64 (any database limits will still apply).
// If the column type is not variable length, such as an int, or if not supported
// by the driver ok is false.
func (ci *ColumnType) Length() (length int64, ok bool) {
	return ci.length, ci.hasLength
}

// DecimalSize returns the scale and precision of a decimal type.
// If not applicable or if not supported ok is false.
func (ci *ColumnType) DecimalSize() (precision, scale int64, ok bool) {
	return ci.precision, ci.scale, ci.hasPrecisionScale
}

// ScanType returns a Go type suitable for scanning into using Rows.Scan.
// If a driver does not support this property ScanType will return
// the type of an empty interface.
func (ci *ColumnType) ScanType() reflect.Type {
	return ci.scanType
}

// Nullable reports whether the column may be null.
// If a driver does not support this property ok will be false.
func (ci *ColumnType) Nullable() (nullable, ok bool) {
	return ci.nullable, ci.hasNullable
}

// DatabaseTypeName returns the database system name of the column type. If an empty
// string is returned, then the driver type name is not supported.
// Consult your driver documentation for a list of driver data types. Length specifiers
// are not included.
// Common type names include "VARCHAR", "TEXT", "NVARCHAR", "DECIMAL", "BOOL",
// "INT", and "BIGINT".
func (ci *ColumnType) DatabaseTypeName() string {
	return ci.databaseType
}

func rowsColumnInfoSetupConnLocked(rowsi driver.Rows) []*ColumnType {
	names := rowsi.Columns()

	list := make([]*ColumnType, len(names))
	for i := range list {
		ci := &ColumnType{
			name: names[i],
		}
		list[i] = ci

		if prop, ok := rowsi.(driver.RowsColumnTypeScanType); ok {
			ci.scanType = prop.ColumnTypeScanType(i)
		} else {
			ci.scanType = reflect.TypeOf(new(any)).Elem()
		}
		if prop, ok := rowsi.(driver.RowsColumnTypeDatabaseTypeName); ok {
			ci.databaseType = prop.ColumnTypeDatabaseTypeName(i)
		}
		if prop, ok := rowsi.(driver.RowsColumnTypeLength); ok {
			ci.length, ci.hasLength = prop.ColumnTypeLength(i)
		}
		if prop, ok := rowsi.(driver.RowsColumnTypeNullable); ok {
			ci.nullable, ci.hasNullable = prop.ColumnTypeNullable(i)
		}
		if prop, ok := rowsi.(driver.RowsColumnTypePrecisionScale); ok {
			ci.precision, ci.scale, ci.hasPrecisionScale = prop.ColumnTypePrecisionScale(i)
		}
	}
	return list
}
