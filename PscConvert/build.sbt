name := "PscConvert"

version := "1.0"

scalaVersion := "2.10.2"

scalacOptions ++= Seq("-target:jvm-1.7", "-deprecation", "-feature", "-unchecked")

fork := true

outputStrategy := Some(StdoutOutput)

connectInput in run := true
