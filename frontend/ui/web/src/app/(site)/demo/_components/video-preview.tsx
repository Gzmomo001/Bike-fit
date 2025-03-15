import DemoContext from "../hooks/createDemoContext";
import { useContext } from "react";

export default function VideoPreview() {
  const {
    selectedFiles: [videoFile],
  } = useContext(DemoContext)!;
  return (
    <>
      <video
        className="w-[400px] max-h-[85%] lg:max-h-[75%] xl:max-h-[50%] sm:w-[600px] lg:w-[800px] rounded-2xl bg-black dark:bg-white shadow-2xl  transition-all duration-1000 "
        controls
      >
        <source src={URL.createObjectURL(videoFile!)} type={videoFile!.type} />
        Your browser does not support the video tag.
      </video>
    </>
  );
}
