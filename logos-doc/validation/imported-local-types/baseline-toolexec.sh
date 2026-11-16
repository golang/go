#!/bin/sh
task_tool=$1
shift
case "$task_tool" in
  */compile) exec /dev/shm/logos-dogfood-93e0563596d1/implicit-type-ice/baseline/compile "$@" ;;
  *) exec "$task_tool" "$@" ;;
esac
