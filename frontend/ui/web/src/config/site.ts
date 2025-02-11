import type { FooterItem, MainNavItem } from "@/types";

export type SiteConfig = typeof siteConfig;

const links = {
  x: "https://twitter.com",
  github: "https://github.com/Gzmomo001/Bike-fit",
  githubAccount: "https://github.com/RobustAI-Lab",
};

export const siteConfig = {
  name: "Bike Fit",
  description:
    "Bike-fit是一个创新的自行车体态拟合系统，旨在帮助骑行者获得最佳的骑行姿势和自行车设置。通过先进的计算机视觉技术和专业的骑行知识，为用户提供个性化的自行车调整建议。",
  url: "https://bikefit.robustai.dev",
  ogImage: "/images/icons8-magic-wand-emoji-32.png",
  links,
  mainNav: [
    {
      title: "主页",
      items: [
        {
          title: "主页",
          href: "/dashboard",
          description: "我的主页",
          items: [],
        },
        {
          title: "抢先体验",
          href: "/demo",
          description: "抢先体验",
          items: [],
        },
      ],
    },
    {
      title: "文档",
      items: [
        {
          title: "API",
          href: "/api",
          description: "API 文档.",
          items: [],
        },
        {
          title: "Guides",
          href: "/guides",
          description: "教程及指引.",
          items: [],
        },
        {
          title: "FAQ",
          href: "/faq",
          description: "常见问题.",
          items: [],
        },
      ],
    },
  ] satisfies MainNavItem[],
  footerNav: [
    {
      title: "贡献",
      items: [
        {
          title: "RobustAI Lab",
          href: "https://robustai.dev",
          external: true,
        },
      ],
    },
    {
      title: "我们的团队",
      items: [
        {
          title: "关于",
          href: "/about",
          external: false,
        },
        {
          title: "联系",
          href: "/contact",
          external: false,
        },
        {
          title: "条款",
          href: "/terms",
          external: false,
        },
        {
          title: "隐私",
          href: "/privacy",
          external: false,
        },
      ],
    },
    {
      title: "社交矩阵",
      items: [
        {
          title: "X",
          href: links.x,
          external: true,
        },
        {
          title: "GitHub",
          href: links.githubAccount,
          external: true,
        },
      ],
    },
  ] satisfies FooterItem[],
};
