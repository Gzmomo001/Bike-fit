"use client";

import { createContext } from "react";

interface DemoContextProps {
  selectedFiles: [
    selectedFiles: File | null,
    setSelectedFiles: (file: File) => void,
  ];
}

const DemoContext = createContext<DemoContextProps | null>(null);
export default DemoContext;
