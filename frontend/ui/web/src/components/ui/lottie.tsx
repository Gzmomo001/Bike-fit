"use client";
/* eslint-disable */
import dynamic from "next/dynamic";
const LottieComponent = dynamic(() => import("lottie-react"), { ssr: false });

interface LottieProps {
  animationData: any;
  options?: {
    loop?: boolean;
    autoplay?: boolean;
    rendererSettings?: {
      preserveAspectRatio?: string;
    };
  };
  className?: string;
}

const Lottie: React.FC<LottieProps> = ({
  animationData,
  options,
  className,
}) => {
  return (
    <LottieComponent
      className={className}
      animationData={animationData}
      loop={options?.loop}
      autoplay={options?.autoplay}
      rendererSettings={options?.rendererSettings}
    />
  );
};

Lottie.displayName = "Lottie";

export default Lottie;
