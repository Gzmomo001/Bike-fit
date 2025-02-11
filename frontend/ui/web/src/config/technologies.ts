import { Icons } from "@/components/icons";

type Technology = {
  id: string;
  name: string;
  icon: keyof typeof Icons;
  description: string;
};

const technologies: Technology[] = [
  {
    id: "MediaPipe",
    name: "MediaPipe",
    icon: "mediapipe",
    description:
      "MediaPipe 是一个轻量级跨平台的机器学习管道，用于处理媒体数据，例如图像、视频和音频。它基于 TensorFlow Lite，可以在移动设备上运行实时机器学习推断。",
  },
];

export { technologies };
export type { Technology };
