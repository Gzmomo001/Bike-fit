'use client';

import { useUser } from '@clerk/nextjs';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from 'sonner';
import Image from 'next/image';
import { useState } from 'react';

const userSchema = z.object({
  firstName: z.string().min(1, '请输入名字'),
  lastName: z.string().min(1, '请输入姓氏'),
  email: z.string().email('请输入有效的邮箱地址'),
});

type UserFormData = z.infer<typeof userSchema>;

export default function Dashboard() {
  const { isLoaded, user } = useUser();
  const [isEditing, setIsEditing] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
    defaultValues: {
      firstName: user?.firstName || '',
      lastName: user?.lastName || '',
      email: user?.emailAddresses[0]?.emailAddress || '',
    },
  });

  const onSubmit = async (data: UserFormData) => {
    if (!user) return;

    try {
      await user.update({
        firstName: data.firstName,
        lastName: data.lastName,
      });

      if (data.email !== user.emailAddresses[0]?.emailAddress) {
        await user.createEmailAddress({ email: data.email });
      }

      toast.success('个人信息更新成功！');
      setIsEditing(false);
    } catch (err) {
      toast.error('更新失败，请重试');
      console.error(err);
    }
  };

  const handleAvatarUpdate = async (file: File) => {
    if (!user) return;

    try {
      await user.setProfileImage({ file });
      toast.success('头像更新成功！');
    } catch (err) {
      toast.error('头像更新失败，请重试');
      console.error(err);
    }
  };

  if (!isLoaded) {
    return <div>加载中...</div>;
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">用户主页</h1>

      <div className="flex items-center space-x-4 mb-6">
        <div className="relative w-20 h-20 rounded-full overflow-hidden">
          <Image
            src={user?.imageUrl || '/default-avatar.png'}
            alt="用户头像"
            fill
            className="object-cover"
          />
        </div>
        <div>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              if (e.target.files?.[0]) {
                handleAvatarUpdate(e.target.files[0]);
              }
            }}
            className="hidden"
            id="avatar-upload"
          />
          <label
            htmlFor="avatar-upload"
            className="text-sm text-blue-600 cursor-pointer"
          >
            更换头像
          </label>
        </div>
      </div>

      {!isEditing && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">名字</label>
            <p className="p-2 bg-gray-100 rounded min-h-4">{user?.firstName}</p>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">姓氏</label>
            <p className="p-2 bg-gray-100 rounded min-h-4">{user?.lastName}</p>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">邮箱</label>
            <p className="p-2 bg-gray-100 rounded">
              {user?.emailAddresses[0]?.emailAddress}
            </p>
          </div>

          <Button onClick={() => setIsEditing(true)} className="w-full">
            修改信息
          </Button>
        </div>
      )}

      {isEditing && (
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">名字</label>
            <Input
              {...register('firstName')}
              placeholder="请输入名字"
              defaultValue={user?.firstName || ''}
            />
            {errors.firstName && (
              <p className="text-sm text-red-500">{errors.firstName.message}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">姓氏</label>
            <Input
              {...register('lastName')}
              placeholder="请输入姓氏"
              defaultValue={user?.lastName || ''}
            />
            {errors.lastName && (
              <p className="text-sm text-red-500">{errors.lastName.message}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">邮箱</label>
            <Input
              {...register('email')}
              placeholder="请输入邮箱"
              defaultValue={user?.emailAddresses[0]?.emailAddress || ''}
            />
            {errors.email && (
              <p className="text-sm text-red-500">{errors.email.message}</p>
            )}
          </div>

          <div className="flex space-x-2">
            <Button type="submit" className="flex-1">
              保存
            </Button>
            <Button
              type="button"
              variant="outline"
              className="flex-1"
              onClick={() => setIsEditing(false)}
            >
              取消
            </Button>
          </div>
        </form>
      )}
    </div>
  );
}