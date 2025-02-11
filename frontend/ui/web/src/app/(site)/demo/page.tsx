import DemoSkeleton from "./_components/demo-skeleton";
import DemoContexProvider from "./hooks/demoContext";

export default function Demo() {
  return (
    <DemoContexProvider>
      <DemoSkeleton />
    </DemoContexProvider>
  );
}
