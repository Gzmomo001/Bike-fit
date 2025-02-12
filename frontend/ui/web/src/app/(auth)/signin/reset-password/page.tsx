import { type Metadata } from "next";
import { env } from "@/env.js";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Shell } from "@/components/shell";
import { ResetPasswordForm } from "@/app/(auth)/_components/reset-password-form";

export const metadata: Metadata = {
  metadataBase: new URL(env.NEXT_PUBLIC_APP_URL),
  title: "Reset Password",
  description: "Enter your email to reset your password",
};

export default function ResetPasswordPage() {
  return (
    <Shell className="max-w-lg">
      <Card>
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl">重制密码</CardTitle>
          <CardDescription>
            输入你的邮箱地址以重制密码
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResetPasswordForm />
        </CardContent>
      </Card>
    </Shell>
  );
}
