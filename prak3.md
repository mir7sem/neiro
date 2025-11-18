# Практическая работа №1
## Этап 1. Первичная разметка
Для проверки имплементации модуля нейропроцессора были составлены определённые файлы, необходимые для реализации интерфейса.
### IP-ядро UART
В разделе IP catalog был выбран AXI UARTLite, затем в выбранном IP-ядре выставлены конкретные настройки:
- AXI CLK Frequency: 100 MHz.
- Baud Rate: 115200.
- Data Bits: 8.
- Parity: None.
### Файл проектных ограничений
```
# Системный клок 100 МГц
set_property -dict { PACKAGE_PIN E3    IOSTANDARD LVCMOS33 } [get_ports { clk }];
create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} -add [get_ports { clk }];

# Кнопка сброса (CPU_RESET)
set_property -dict { PACKAGE_PIN C12   IOSTANDARD LVCMOS33 } [get_ports { rst }];

# UART (подключен к USB-UART мосту)
set_property -dict { PACKAGE_PIN C4    IOSTANDARD LVCMOS33 } [get_ports { uart_rxd }];
set_property -dict { PACKAGE_PIN D4    IOSTANDARD LVCMOS33 } [get_ports { uart_txd }];
```
### Модуль верхнего уровня
```verilog
`timescale 1ns / 1ps

module top_module (
    input  wire        clk,       // Системный клок (100МГц для Nexys A7)
    input  wire        rst,       // Кнопка сброса
    input  wire        uart_rxd,  // UART RX с FTDI чипа
    output wire        uart_txd   // UART TX на FTDI чип
);

    // ========================================
    // Параметры
    // ========================================
    localparam SCALE = 100000;

    // FSM состояния
    localparam [4:0] S_IDLE              = 5'd0;
    localparam [4:0] S_RX_CHECK          = 5'd1; // Проверить, есть ли данные в RX FIFO
    localparam [4:0] S_RX_READ           = 5'd2; // Прочитать байт из RX FIFO
    localparam [4:0] S_RX_ASSEMBLE       = 5'd3; // Собрать 32-битное слово
    localparam [4:0] S_START_NN          = 5'd4;
    localparam [4:0] S_WAIT_NN           = 5'd5;
    localparam [4:0] S_LATCH_RESULT      = 5'd6;
    localparam [4:0] S_TX_CHECK          = 5'd7; // Проверить, есть ли место в TX FIFO
    localparam [4:0] S_TX_WRITE          = 5'd8; // Записать байт в TX FIFO

    reg [4:0] state = S_IDLE;

    // Регистры для данных
    reg signed [31:0] coords [0:5]; // x1, y1, x2, y2, x3, y3
    reg signed [31:0] m_result, b_result;

    // Счетчики
    reg [4:0] rx_byte_count = 0; // 0-23
    reg [2:0] rx_word_count = 0; // 0-5
    reg [3:0] tx_byte_count = 0; // 0-7
    
    // Временный регистр для сборки байтов
    reg [31:0] byte_assembler;

    // Сигналы для нейросети
    reg nn_start = 1'b0;
    wire nn_valid_out;
    wire signed [31:0] nn_m_out;
    wire signed [31:0] nn_b_out;
    
    // ========================================
    // AXI Lite сигналы для UART
    // ========================================
    // Адреса регистров в AXI UART Lite
    localparam AXI_RX_FIFO_ADDR = 4'h00;
    localparam AXI_TX_FIFO_ADDR = 4'h04;
    localparam AXI_STAT_REG_ADDR= 4'h08;
    localparam AXI_CTRL_REG_ADDR= 4'h0C;

    // Биты в статус-регистре
    localparam RX_VALID_BIT = 0; // Data in RX FIFO
    localparam TX_FULL_BIT  = 3; // TX FIFO is full

    reg  s_axi_arvalid = 1'b0;
    reg  s_axi_awvalid = 1'b0;
    reg  s_axi_wvalid = 1'b0;
    reg  s_axi_rready = 1'b0;
    reg  s_axi_bready = 1'b0;
    reg  [3:0] s_axi_araddr;
    reg  [3:0] s_axi_awaddr;
    reg  [31:0] s_axi_wdata;
    
    wire s_axi_arready;
    wire s_axi_awready;
    wire s_axi_wready;
    wire s_axi_rvalid;
    wire [31:0] s_axi_rdata;

    // ========================================
    // Инстанцирование Нейропроцессора
    // ========================================
    neural_inference #(
        .SCALE(SCALE)
    ) nn_inst (
        .clk(clk),
        .rst(rst),
        .start(nn_start),
        .x1(coords[0]), .y1(coords[1]),
        .x2(coords[2]), .y2(coords[3]),
        .x3(coords[4]), .y3(coords[5]),
        .m_out(nn_m_out),
        .b_out(nn_b_out),
        .valid_out(nn_valid_out)
    );

    // ========================================
    // Инстанцирование UART IP-ядра
    // ========================================
    axi_uartlite_0 uart_inst (
        .s_axi_aclk(clk),
        .s_axi_aresetn(~rst),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(4'b0001),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(),
        .s_axi_bvalid(),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .rx(uart_rxd),
        .tx(uart_txd)
    );

    // ========================================
    // Основной конечный автомат (FSM)
    // ========================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            rx_byte_count <= 0;
            rx_word_count <= 0;
            tx_byte_count <= 0;
            nn_start <= 1'b0;
            s_axi_arvalid <= 1'b0;
            s_axi_awvalid <= 1'b0;
            s_axi_wvalid <= 1'b0;
            s_axi_rready <= 1'b0;
            s_axi_bready <= 1'b0;
        end else begin
            // Сбрасываем управляющие сигналы по умолчанию
            nn_start <= 1'b0;
            s_axi_arvalid <= 1'b0;
            s_axi_awvalid <= 1'b0;
            s_axi_wvalid <= 1'b0;
            s_axi_rready <= 1'b0;
            s_axi_bready <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    rx_byte_count <= 0;
                    rx_word_count <= 0;
                    tx_byte_count <= 0;
                    state <= S_RX_CHECK;
                end
                
                // --- Прием данных ---
                S_RX_CHECK: begin
                    s_axi_araddr <= AXI_STAT_REG_ADDR;
                    s_axi_arvalid <= 1'b1;
                    if (s_axi_arready) begin
                        s_axi_rready <= 1'b1;
                        if (s_axi_rvalid && s_axi_rdata[RX_VALID_BIT]) begin
                            state <= S_RX_READ;
                        end
                    end
                end
                
                S_RX_READ: begin
                    s_axi_araddr <= AXI_RX_FIFO_ADDR;
                    s_axi_arvalid <= 1'b1;
                    if (s_axi_arready) begin
                        s_axi_rready <= 1'b1;
                        if (s_axi_rvalid) begin
                            byte_assembler <= {byte_assembler[23:0], s_axi_rdata[7:0]};
                            state <= S_RX_ASSEMBLE;
                        end
                    end
                end
                
                S_RX_ASSEMBLE: begin
                    if (rx_byte_count[1:0] == 2'b11) begin // Каждый 4-й байт
                        coords[rx_word_count] <= {s_axi_rdata[7:0], byte_assembler[23:0]};
                        rx_word_count <= rx_word_count + 1;
                    end
                    
                    if (rx_byte_count == 23) begin
                        state <= S_START_NN;
                    end else begin
                        rx_byte_count <= rx_byte_count + 1;
                        state <= S_RX_CHECK;
                    end
                end

                // --- Запуск и ожидание Нейросети ---
                S_START_NN: begin
                    nn_start <= 1'b1;
                    state <= S_WAIT_NN;
                end
                
                S_WAIT_NN: begin
                    if (nn_valid_out) begin
                        state <= S_LATCH_RESULT;
                    end
                end

                S_LATCH_RESULT: begin
                    m_result <= nn_m_out;
                    b_result <= nn_b_out;
                    state <= S_TX_CHECK;
                end

                // --- Отправка данных ---
                S_TX_CHECK: begin
                    s_axi_araddr <= AXI_STAT_REG_ADDR;
                    s_axi_arvalid <= 1'b1;
                    if (s_axi_arready) begin
                        s_axi_rready <= 1'b1;
                        if (s_axi_rvalid && !s_axi_rdata[TX_FULL_BIT]) begin
                           state <= S_TX_WRITE;
                        end
                    end
                end
                
                S_TX_WRITE: begin
                    s_axi_awaddr <= AXI_TX_FIFO_ADDR;
                    s_axi_wdata[7:0] <= (tx_byte_count < 4) ? m_result >> (tx_byte_count[1:0] * 8) : b_result >> (tx_byte_count[1:0] * 8);
                    s_axi_awvalid <= 1'b1;
                    s_axi_wvalid <= 1'b1;
                    
                    if (s_axi_awready && s_axi_wready) begin
                        if (tx_byte_count == 7) begin
                            state <= S_IDLE; // Все отправили, ждем новых данных
                        end else begin
                            tx_byte_count <= tx_byte_count + 1;
                            state <= S_TX_CHECK;
                        end
                    end
                end
            endcase
        end
    end
endmodule
```
