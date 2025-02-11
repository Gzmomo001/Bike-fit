"use client";

import FileUploader from "@/components/file-uploader";
import { useContext } from "react";
import DemoContext from "../hooks/createDemoContext";
import VideoPreview from "./video-preview";

export default function DemoSkeleton() {
  const {
    selectedFiles: [selectedFile, setSelectedFile],
  } = useContext(DemoContext)!;

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      {!selectedFile && (
        <FileUploader
          selectedFile={selectedFile}
          setSelectedFile={setSelectedFile}
        />
      )}
      {selectedFile && <VideoPreview />}
    </div>
  );
}
