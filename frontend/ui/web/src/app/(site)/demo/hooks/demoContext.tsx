"use client";
import { useState } from "react";
import DemoContext from "./createDemoContext";

export default function DemoContexProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [selectedFiles, setSelectedFiles] = useState<File | null>(null);

  return (
    <DemoContext.Provider
      value={{ selectedFiles: [selectedFiles, setSelectedFiles] }}
    >
      {children}
    </DemoContext.Provider>
  );
}
