import Link from "next/link";

import { Icons } from "@/components/icons";
import { Shell } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { siteConfig } from "@/config/site";
import { type getGithubStars } from "@/lib/actions/github";

import { ContentSection } from "@/components/content-section";
import { Technologies } from "@/components/technologies";
import { ExternalLinkIcon } from "@radix-ui/react-icons";

interface MainProps {
  githubStarsPromise: ReturnType<typeof getGithubStars>;
}

export async function Main({ githubStarsPromise }: MainProps) {
  // See the "Parallel data fetching" docs: https://nextjs.org/docs/app/building-your-application/data-fetching/patterns#parallel-data-fetching
  const githubStars = Promise.resolve(githubStarsPromise);

  return (
    <Shell className="max-w-6xl">
      <section className="mx-auto flex w-full max-w-5xl flex-col items-center justify-center gap-4 py-24 text-center md:py-32">
        <div
          className="flex animate-fade-up flex-col space-y-2"
          style={{ animationDelay: "0.10s", animationFillMode: "both" }}
        >
          <Link href={siteConfig.links.github} target="_blank" rel="noreferrer">
            <Badge
              aria-hidden="true"
              className="rounded-full px-3.5 py-1.5"
              variant="secondary"
            >
              <Icons.github className="mr-2 size-3.5" aria-hidden="true" />
              {githubStars} 🌟
            </Badge>
            <span className="sr-only">GitHub</span>
          </Link>
        </div>
        <h1
          className="animate-fade-up text-balance font-heading text-3xl sm:text-5xl md:text-6xl lg:text-7xl"
          style={{ animationDelay: "0.20s", animationFillMode: "both" }}
        >
          开启你的 <span className="text-primary">骑行每一天</span>
        </h1>
        <p
          className="max-w-2xl animate-fade-up text-balance leading-normal text-muted-foreground sm:text-xl sm:leading-8"
          style={{ animationDelay: "0.30s", animationFillMode: "both" }}
        >
          Bike-fit是一个创新的自行车体态拟合系统，旨在帮助骑行者获得最佳的骑行姿势和自行车设置。通过先进的计算机视觉技术和专业的骑行知识，为用户提供个性化的自行车调整建议。
        </p>
        <div
          className="flex animate-fade-up flex-wrap items-center justify-center gap-4"
          style={{ animationDelay: "0.40s", animationFillMode: "both" }}
        >
          <Button asChild>
            <Link
              href={siteConfig.links.github}
              target="_blank"
              className="cursor-pointer flex items-center"
            >
              现在克隆{" "}
              <ExternalLinkIcon className="ml-2 size-4" aria-hidden="true" />
              <span className="sr-only">现在克隆</span>
            </Link>
          </Button>
          <Button variant="outline" asChild>
            <Link href="/demo">
              抢先体验
              <span className="sr-only">抢先体验</span>
            </Link>
          </Button>
        </div>
      </section>

      <ContentSection
        title="技术栈"
        description="我们的使用最前沿的技术，来帮助您获得最佳的骑行体验。"
        linkText="查看所有"
        className="pt-8 md:py-10 lg:py-12"
        href="/tech"
        asChild
      >
        <Technologies />
      </ContentSection>
    </Shell>
  );
}
