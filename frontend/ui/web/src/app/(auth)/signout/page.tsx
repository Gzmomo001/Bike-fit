import type { Metadata } from "next";
import { env } from "@/env.js";

import {
  PageHeader,
  PageHeaderDescription,
  PageHeaderHeading,
} from "@/components/page-header";
import { Shell } from "@/components/shell";
import { LogOutButtons } from "@/app/(auth)/_components/logout-buttons";

export const metadata: Metadata = {
  metadataBase: new URL(env.NEXT_PUBLIC_APP_URL),
  title: "Sign out",
  description: "Sign out of your account",
};

export default function SignOutPage() {
  return (
    <Shell className="max-w-md">
      <PageHeader className="text-center">
        <PageHeaderHeading size="sm">退出</PageHeaderHeading>
        <PageHeaderDescription size="sm">确定要退出吗？</PageHeaderDescription>
      </PageHeader>
      <LogOutButtons />
    </Shell>
  );
}
