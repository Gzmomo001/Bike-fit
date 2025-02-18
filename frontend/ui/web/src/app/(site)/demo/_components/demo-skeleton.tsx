"use client";

import FileUploader from "@/components/file-uploader";
import { useContext, useEffect, useRef, useState } from "react";
import DemoContext from "../hooks/createDemoContext";
import VideoPreview from "./video-preview";
import ReactMarkdown from "react-markdown";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import { Separator } from "@/components/ui/separator";

import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  HoverCard,
  HoverCardTrigger,
  HoverCardContent,
} from "@/components/ui/hover-card";
import { Button } from "@/components/ui/button";
import { Icons } from "@/components/icons";
import { cn } from "@/lib/utils";

export default function DemoSkeleton() {
  const {
    selectedFiles: [selectedFile, setSelectedFile],
  } = useContext(DemoContext)!;
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [info, setInfo] = useState<any>(null);
  const [reasoning, setReasoning] = useState("");
  const [response, setResponse] = useState("");

  const handleFileUpload = async (file: File) => {
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append("video", file);

    try {
      const response = await fetch("http://localhost:8000/api/analyze/video", {
        method: "POST",
        body: formData,
      });

      if (!response.body) {
        throw new Error("ReadableStream not yet supported in this browser.");
      }

      setIsAnalyzing(false);

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        const chunk = decoder.decode(value, { stream: true });
        const chunks = chunk.split("\n");
        try {
          chunks.forEach((chunk) => {
            if (chunk.trim()) {
              const json = JSON.parse(chunk);
              console.log(json);
              switch (json.type) {
                case "info":
                  setInfo(json.message);
                  break;
                case "reasoning":
                  setReasoning((prev) => prev + json.message);
                  break;
                case "response":
                  setResponse((prev) => prev + json.message);
                  break;
              }
            }
          });
        } catch (e) {
          console.log("error", e);
        }
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    if (selectedFile) {
      handleFileUpload(selectedFile);
    }
  }, [selectedFile]);

  const renderMessage = (message: string) => {
    return <ReactMarkdown>{message}</ReactMarkdown>;
  };

  return (
    <div
      className={cn(
        "p-4 flex justify-center place-items-center items-center min-h-screen relative",
        selectedFile &&
          "grid grid-rows-2 grid-cols-1 lg:gap-4 lg:grid-rows-1 lg:grid-cols-2",
      )}
    >
      {!selectedFile && (
        <FileUploader
          selectedFile={selectedFile}
          setSelectedFile={setSelectedFile}
        />
      )}
      {selectedFile && <VideoPreview />}
      {selectedFile && (
        <ScrollArea className="h-72 max-w-full w-[370px] xs:w-[450px] md:w-[550px] lg:w-[700px] xxs:hover:h-[450px] hover:w-full  lg:hover:h-[700px] transition-all ease-in-out duration-1000 p-4 rounded-xl border">
          <div>
            {!reasoning && !response && (
              <TextGenerateEffect
                duration={2}
                words="正在思考..."
                className="text-foreground font-serif text-md"
              />
            )}
            {reasoning && response && (
              <TextGenerateEffect
                duration={2}
                words="思考完成"
                className="text-foreground font-serif text-md"
              />
            )}
            {reasoning && <Separator />}
            <div className="text-foreground font-serif text-sm">
              {renderMessage(reasoning)}
            </div>
            {response && (
              <TextGenerateEffect
                duration={2}
                words="我的建议"
                className="text-foreground font-serif text-md"
              />
            )}
            {response && <Separator />}

            <div>{renderMessage(response)}</div>
          </div>
        </ScrollArea>
      )}
      {selectedFile && (
        <HoverCard>
          <HoverCardTrigger asChild>
            <Button
              variant={"outline"}
              className={cn(
                "absolute top-4 left-4",
                !info && "cursor-not-allowed",
              )}
            >
              {!info && <Icons.spinner className="size-8 animate-spin" />}
              {info && "查看量化指标"}
            </Button>
          </HoverCardTrigger>
          {info && (
            <HoverCardContent className="w-[400px]">
              <Table>
                <TableCaption>你的量化指标</TableCaption>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[100px]">量化指标</TableHead>
                    <TableHead className="w-[100px]">值</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(info).map(([key, value]) => (
                    <TableRow key={key}>
                      <TableCell className="">{key}</TableCell>
                      <TableCell className="">
                        {(value as number).toFixed(2)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </HoverCardContent>
          )}
        </HoverCard>
      )}
    </div>
  );
}
