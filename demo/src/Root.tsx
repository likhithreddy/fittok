import React from "react";
import { Composition } from "remotion";
import { FittokDemo } from "./FittokDemo";

export const Root: React.FC = () => (
  <Composition
    id="FittokDemo"
    component={FittokDemo}
    durationInFrames={2370}
    fps={30}
    width={1280}
    height={720}
    defaultProps={{}}
  />
);
