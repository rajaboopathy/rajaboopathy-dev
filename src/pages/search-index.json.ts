import { getCollection } from 'astro:content';

export async function GET() {
  const blogPosts = await getCollection('blog');
  const gardenPosts = await getCollection('garden');

  const items = [
    ...blogPosts.map(post => ({
      id: post.id,
      type: 'blog',
      title: post.data.title,
      description: post.data.description,
      tags: post.data.tags,
      category: post.data.category,
      url: `${import.meta.env.BASE_URL}blog/${post.id}`
    })),
    ...gardenPosts.map(post => ({
      id: post.id,
      type: 'garden',
      title: post.data.title,
      description: post.data.description,
      tags: post.data.tags,
      category: post.data.category,
      url: `${import.meta.env.BASE_URL}garden/${post.id}`
    }))
  ];

  return new Response(JSON.stringify(items), {
    headers: { 'Content-Type': 'application/json' }
  });
}
