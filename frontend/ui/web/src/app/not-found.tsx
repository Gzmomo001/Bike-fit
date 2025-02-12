import Image from "next/image";
export default function NotFound() {
  return (
    <div className="flex items-center justify-center h-screen">
      <div className="text-center">
        <div className="flex gap-8">
          <Image alt="not found" src={"/icon.png"} width={100} height={100} />
          <div className="flex flex-col items-center justify-center">
            <h1 className="text-6xl font-bold text-gray-800">404</h1>
            <p className="text-2lg text-gray-600">你已骑行到世界的荒漠</p>
          </div>
        </div>
      </div>
    </div>
  );
}
