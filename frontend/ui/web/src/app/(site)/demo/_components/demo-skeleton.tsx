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
  const [gifData, setGifData] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setIsAnalyzing(true);
    setGifData(null); // 重置之前的GIF数据
    setInfo(null); // 重置之前的量化指标
    setReasoning(""); // 重置之前的推理数据
    setResponse(""); // 重置之前的响应数据
    const formData = new FormData();
    formData.append("video", file);
    console.log("开始上传视频文件...");

    try {
      // 显示上传进度提示
      setResponse("正在处理视频，请稍候...");
      
      const response = await fetch("http://localhost:8000/api/analyze/video", {
        method: "POST",
        body: formData,
      });
      console.log("收到服务器响应:", response.status, response.statusText);

      if (!response.body) {
        throw new Error("ReadableStream not yet supported in this browser.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;
      console.log("开始读取流式响应...");

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        
        if (done) {
          console.log("流式响应读取完成");
          break;
        }
        
        const chunk = decoder.decode(value, { stream: true });
        console.log("收到数据块:", chunk.length, "字节");
        const chunks = chunk.split("\n");
        console.log("分割后的数据块数量:", chunks.length);
        
        try {
          chunks.forEach((chunk) => {
            if (chunk.trim()) {
              console.log("处理数据块:", chunk.substring(0, 50) + "...");
              try {
                const json = JSON.parse(chunk);
                console.log("解析的JSON数据类型:", json.type);
                
                switch (json.type) {
                  case "gif":
                    console.log("收到GIF数据，长度:", json.data.length);
                    if (json.data && typeof json.data === 'string') {
                      // 验证base64数据
                      try {
                        // 尝试解码一小部分base64数据，验证其有效性
                        const testDecode = atob(json.data.slice(0, 100));
                        console.log("GIF数据有效，设置GIF数据");
                        setGifData(json.data);
                        // 确保分析状态被设置为完成
                        setIsAnalyzing(false);
                      } catch (e) {
                        console.error("无效的base64 GIF数据:", e);
                      }
                    } else {
                      console.error("无效的GIF数据格式:", typeof json.data);
                    }
                    break;
                  case "info":
                    console.log("收到测量信息:", JSON.stringify(json.message));
                    if (json.message && typeof json.message === 'object') {
                      console.log("测量信息是对象格式:", JSON.stringify(json.message));
                      // 确保测量数据正确设置
                      setInfo(json.message);
                      console.log("已设置info状态:", JSON.stringify(json.message));
                    } else if (typeof json.message === 'string') {
                      try {
                        const parsedInfo = JSON.parse(json.message);
                        console.log("解析后的测量信息:", JSON.stringify(parsedInfo));
                        setInfo(parsedInfo);
                        console.log("已设置info状态(从字符串解析):", JSON.stringify(parsedInfo));
                      } catch (e) {
                        console.error("无法解析测量信息字符串:", e);
                        // 如果不能解析为JSON，直接将字符串作为单个字段显示
                        console.log("将字符串作为单个测量结果处理");
                        const singleValueInfo = { "测量结果": json.message };
                        setInfo(singleValueInfo);
                        console.log("已设置info状态(字符串):", JSON.stringify(singleValueInfo));
                      }
                    } else {
                      console.error("无效的测量信息格式:", typeof json.message);
                      const errorInfo = { "错误": "无效的测量数据格式" };
                      setInfo(errorInfo);
                      console.log("已设置info状态(错误):", JSON.stringify(errorInfo));
                    }
                    break;
                  case "reasoning":
                    console.log("收到推理数据");
                    setReasoning((prev: string) => prev + json.message);
                    break;
                  case "response":
                    console.log("收到响应数据");
                    setResponse((prev: string) => prev + json.message);
                    break;
                  case "error":
                    console.error("收到错误信息:", json.message);
                    // 显示错误信息
                    setResponse((prev: string) => prev + "\n**错误:** " + json.message);
                    break;
                  default:
                    console.warn("未知的数据类型:", json.type);
                    break;
                }
              } catch (parseError) {
                console.error("解析JSON数据失败:", parseError, "原始数据:", chunk);
              }
            }
          });
        } catch (e) {
          console.error("处理数据块时出错:", e);
        }
      }
    } catch (error) {
      console.error("上传文件时出错:", error);
      setResponse((prev: string) => prev + "\n**错误:** 处理视频时发生错误，请重试");
    } finally {
      // 确保无论如何都设置分析状态为完成
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
          "grid grid-rows-2 grid-cols-1 lg:gap-8 lg:grid-rows-1 lg:grid-cols-2",
      )}
    >
      {console.log("渲染时的info状态:", info ? JSON.stringify(info) : "null")}
      {!selectedFile && (
        <FileUploader
          selectedFile={selectedFile}
          setSelectedFile={setSelectedFile}
        />
      )}
      
      {/* 左侧区域：显示视频或GIF */}
      {selectedFile && (
        <div className="flex flex-col items-center justify-center h-full w-full max-w-[900px] p-4 relative">
          {/* 处理状态指示器 */}
          {isAnalyzing && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/30 rounded-xl z-10">
              <div className="flex flex-col items-center bg-white p-6 rounded-lg shadow-lg">
                <Icons.spinner className="size-10 animate-spin mb-4" />
                <div className="text-lg font-medium">视频处理中...</div>
                <div className="text-sm text-gray-500 mt-2">请稍候，正在分析姿态</div>
              </div>
            </div>
          )}
          
          {/* 当有GIF数据时显示GIF图像，否则显示原始视频 */}
          {gifData ? (
            <>
              {console.log("渲染GIF图像，数据长度:", gifData.length)}
              <div className="relative w-full flex flex-col items-center">
                <h3 className="text-lg font-semibold mb-2">姿态分析结果</h3>
                <div className="relative w-full bg-gray-100 p-2 rounded-xl">
                  <img 
                    src={`data:image/gif;base64,${gifData}`} 
                    alt="姿态分析GIF" 
                    className="w-full max-w-[800px] mx-auto rounded-xl shadow-lg object-contain border-2 border-blue-500"
                    style={{background: '#f0f0f0', minHeight: '400px'}}
                    onError={(e) => {
                      console.error("GIF图像加载失败:", e);
                      // 显示错误信息
                      e.currentTarget.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeAJ5jYI7QQAAAABJRU5ErkJggg==";
                      e.currentTarget.alt = "图像加载失败 - 请检查控制台";
                      e.currentTarget.style.width = "400px";
                      e.currentTarget.style.height = "300px";
                      e.currentTarget.style.border = "4px solid red";
                      // 在DOM中添加错误信息
                      const errorText = document.createElement('div');
                      errorText.textContent = "图像加载失败，请检查控制台";
                      errorText.style.color = "red";
                      errorText.style.fontSize = "16px";
                      errorText.style.marginTop = "10px";
                      e.currentTarget.parentNode?.appendChild(errorText);
                    }}
                    onLoad={(e) => {
                      console.log("GIF图像加载成功");
                      // 记录图像的实际尺寸
                      console.log("图像尺寸:", (e.currentTarget as HTMLImageElement).naturalWidth, "x", (e.currentTarget as HTMLImageElement).naturalHeight);
                    }}
                  />
                </div>
                
                {/* 显示量化指标 */}
                {info && (
                  <div className="mt-4 w-full max-w-[800px]">
                    <h4 className="text-md font-semibold mb-2">量化指标</h4>
                    <div className="bg-white rounded-lg p-3 shadow-sm border">
                      {Object.keys(info).length > 0 ? (
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                          {Object.entries(info).map(([key, value]) => (
                            <div key={key} className="bg-gray-50 p-2 rounded">
                              <div className="text-xs text-gray-500">{key}</div>
                              <div className="font-medium">
                                {typeof value === 'number' || (typeof value === 'string' && !isNaN(Number(value)))
                                  ? `${typeof value === 'number' ? value : Number(value)}°` 
                                  : typeof value === 'object'
                                    ? JSON.stringify(value)
                                    : String(value)}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-2 text-gray-500">无量化指标数据</div>
                      )}
                    </div>
                  </div>
                )}
                
                <Button 
                  variant="outline" 
                  onClick={() => setGifData(null)}
                  className="mt-4"
                >
                  查看原始视频
                </Button>
              </div>
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <VideoPreview />
            </div>
          )}
        </div>
      )}
      
      {/* 右侧区域：分析结果 */}
      {selectedFile && (
        <div className="flex flex-col h-full w-full p-4">
          <h3 className="text-lg font-semibold mb-2">分析结果</h3>
          <ScrollArea className="flex-grow max-w-full w-full rounded-xl border bg-white p-4 shadow-sm">
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
        </div>
      )}
      {selectedFile && (
        <HoverCard>
          <HoverCardTrigger asChild>
            <Button
              variant={"outline"}
              className={cn(
                "absolute top-4 left-4",
                (!info || Object.keys(info).length === 0) && "cursor-not-allowed",
              )}
            >
              {(!info || Object.keys(info).length === 0) && <Icons.spinner className="size-8 animate-spin" />}
              {info && Object.keys(info).length > 0 && "查看量化指标"}
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
                        {typeof value === 'number' || (typeof value === 'string' && !isNaN(Number(value)))
                          ? `${typeof value === 'number' ? value : Number(value)}°` 
                          : typeof value === 'object'
                            ? JSON.stringify(value)
                            : String(value)}
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
