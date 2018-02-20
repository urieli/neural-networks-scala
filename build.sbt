import scalariform.formatter.preferences._

name := "neural-networks"

version := "1.0"

scalaVersion := "2.12.4"

libraryDependencies  ++= Seq(
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "org.scalatest" %% "scalatest" % "3.0.3" % "test",
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.apache.commons" % "commons-compress" % "1.16.1",
  "org.rogach" %% "scallop" % "3.1.1",
  "io.spray" %%  "spray-json" % "1.3.3",
  "com.beachape" %% "enumeratum" % "1.5.12"
)


resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"

scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked")

scalariformAutoformat := true

scalariformPreferences := {
scalariformPreferences.value
  .setPreference(AlignSingleLineCaseStatements, true)
  .setPreference(DoubleIndentConstructorArguments, true)
  .setPreference(MultilineScaladocCommentsStartOnFirstLine, true)
}

fork in run := true