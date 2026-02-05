FROM node:20-alpine AS build

WORKDIR /app

ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL

COPY apps/web/package.json /app/package.json
COPY apps/web/tsconfig.json /app/tsconfig.json
COPY apps/web/tsconfig.node.json /app/tsconfig.node.json
COPY apps/web/vite.config.ts /app/vite.config.ts

RUN npm install

COPY apps/web/index.html /app/index.html
COPY apps/web/src /app/src

RUN npm run build

FROM nginx:1.27-alpine

COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
