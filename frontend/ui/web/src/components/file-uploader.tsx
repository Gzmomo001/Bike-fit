"use client";

import React, { useRef, useEffect } from "react";
import { toast } from "sonner";

export default function FileUploader({
  selectedFile,
  setSelectedFile,
}: {
  selectedFile: File | null;
  setSelectedFile: (file: File) => void;
}) {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const maxSize = 500 * 1024 * 1024;
      if (file && file.size > maxSize) {
        toast.error("文件大小超过 500MB 限制.");
        event.target.value = "";
        return;
      }
      setSelectedFile(file);
    }
  };

  const dropAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const dropArea = dropAreaRef.current;

    const handleDragEnter = (event: DragEvent) => {
      if (dropArea) {
        dropArea.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
      }
      event.preventDefault();
      event.stopPropagation();
    };

    const handleDragOver = (event: DragEvent) => {
      event.preventDefault();
      event.stopPropagation();
    };

    const handleDrop = (event: DragEvent) => {
      event.preventDefault();
      event.stopPropagation();
      const file = event.dataTransfer?.files?.[0];
      if (file) {
        const maxSize = 500 * 1024 * 1024; // 5MB
        if (file && file.size > maxSize) {
          toast.error("文件大小超过 500MB 限制.");
          return;
        }
        setSelectedFile(file);
      }
    };

    if (dropArea) {
      dropArea.addEventListener("dragenter", handleDragEnter);
      dropArea.addEventListener("dragover", handleDragOver);
      dropArea.addEventListener("drop", handleDrop);
    }

    return () => {
      if (dropArea) {
        dropArea.removeEventListener("dragenter", handleDragEnter);
        dropArea.removeEventListener("dragover", handleDragOver);
        dropArea.removeEventListener("drop", handleDrop);
      }
    };
  }, []);

  return (
    <div
      ref={dropAreaRef}
      className="flex drop-shadow-2xl items-center justify-center w-full h-full p-0"
    >
      <div className="w-full max-w-lg p-8 space-y-3 rounded-xl bg-gray-50 text-gray-800">
        <div className="flex flex-col items-center justify-center w-full">
          <label
            htmlFor="input"
            className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer hover:bg-gray-100"
          >
            <span className="flex flex-col items-center justify-center pt-5 pb-6">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="w-6 h-6 text-gray-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                ></path>
              </svg>
              <span className="font-medium text-gray-600 text-center">
                拖动视频文件到此处或者
                <span className="text-blue-600 underline ml-[4px]">
                  点击浏览
                </span>
              </span>
            </span>
            <input
              type="file"
              name="file_upload"
              className="hidden"
              accept="*"
              id="input"
              onChange={handleFileChange}
            />
          </label>
        </div>
        {/* {selectedFile && (
          <div className="mt-4">
            <p>
              <strong>Selected file:</strong> {selectedFile.name}
            </p>
            <p>
              <strong>Type:</strong> {selectedFile.type}
            </p>
            <p>
              <strong>Size:</strong>{" "}
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
            </p>
            <p>
              <strong>Last Modified:</strong>{" "}
              {new Date(selectedFile.lastModified).toLocaleDateString()}
            </p>
          </div>
        )} */}
      </div>
    </div>
  );
}
