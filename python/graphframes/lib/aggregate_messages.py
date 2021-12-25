#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Callable, cast, Type, TypeVar, TYPE_CHECKING

from pyspark import SparkContext
from pyspark.sql import Column, DataFrame, functions as sqlfunctions

if TYPE_CHECKING:
    from py4j.java_gateway import JavaObject, JVMView # type: ignore[import]


_T = TypeVar("_T")
_U = TypeVar("_U")


def _java_api(jsc: Type[SparkContext]) -> "JavaObject":
    javaClassName = "org.graphframes.GraphFramePythonAPI"
    return (
        cast("JVMView", jsc._jvm)  # type: ignore[attr-defined]
        .Thread.currentThread()
        .getContextClassLoader()
        .loadClass(javaClassName)
        .newInstance()
    )


class _ClassProperty(object):
    """Custom read-only class property descriptor.

    The underlying method should take the class as the sole argument.
    """

    def __init__(self, f: Callable[[_U], _T]):
        self.f = f
        self.__doc__ = f.__doc__

    def __get__(self, instance: Any, owner: _U) -> _T:
        return self.f(owner)


class AggregateMessages(object):
    """Collection of utilities usable with :meth:`graphframes.GraphFrame.aggregateMessages()`."""

    @_ClassProperty
    def src(cls) -> Column:
        """Reference for source column, used for specifying messages."""
        jvm_gf_api = _java_api(SparkContext)
        return sqlfunctions.col(jvm_gf_api.SRC())

    @_ClassProperty
    def dst(cls) -> Column:
        """Reference for destination column, used for specifying messages."""
        jvm_gf_api = _java_api(SparkContext)
        return sqlfunctions.col(jvm_gf_api.DST())

    @_ClassProperty
    def edge(cls) -> Column:
        """Reference for edge column, used for specifying messages."""
        jvm_gf_api = _java_api(SparkContext)
        return sqlfunctions.col(jvm_gf_api.EDGE())

    @_ClassProperty
    def msg(cls) -> Column:
        """Reference for message column, used for specifying aggregation function."""
        jvm_gf_api = _java_api(SparkContext)
        return sqlfunctions.col(jvm_gf_api.aggregateMessages().MSG_COL_NAME())

    @staticmethod
    def getCachedDataFrame(df: DataFrame) -> DataFrame:
        """
        Create a new cached copy of a DataFrame.

        This utility method is useful for iterative DataFrame-based algorithms. See Scala
        documentation for more details.

        WARNING: This is NOT the same as `DataFrame.cache()`.
                 The original DataFrame will NOT be cached.
        """
        sqlContext = df.sql_ctx
        jvm_gf_api = _java_api(sqlContext._sc)  # type: ignore[attr-defined]
        jdf = jvm_gf_api.aggregateMessages().getCachedDataFrame(df._jdf)  # type: ignore[attr-defined]
        return DataFrame(jdf, sqlContext)
