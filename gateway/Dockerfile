FROM golang:1.23-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /out/gateway ./cmd/gateway

FROM gcr.io/distroless/static-debian12
COPY --from=build /out/gateway /gateway
COPY config.example.yaml /etc/reroute/config.yaml
EXPOSE 8080
ENTRYPOINT ["/gateway", "-config", "/etc/reroute/config.yaml"]
