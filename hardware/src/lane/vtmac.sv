// Copyright 2025
// 
// Description:
// This is the T-MAC (Table Lookup for Low-Bit LLM) unit for Ara.
// It implements the table-lookup approach for matrix multiplication
// with bit-serial decomposition as described in the T-MAC paper.

module vtmac import ara_pkg::*; import rvv_pkg::*; import cf_math_pkg::idx_width; #(
    parameter  int    unsigned NrLanes         = 0,
    parameter  int    unsigned VLEN            = 0,
    // Type used to address vector register file elements
    parameter  type            vaddr_t         = logic,
    parameter  type            vfu_operation_t = logic,
    // Dependant parameters. DO NOT CHANGE!
    localparam int    unsigned DataWidth    = $bits(elen_t),
    localparam int    unsigned StrbWidth    = DataWidth/8,
    localparam type            strb_t       = logic [StrbWidth-1:0],
    localparam type            vlen_t       = logic[$clog2(VLEN+1)-1:0]
  ) (
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic[idx_width(NrLanes)-1:0] lane_id_i,
    
    // Interface with the lane sequencer
    input  vfu_operation_t               vfu_operation_i,
    input  logic                         vfu_operation_valid_i,
    output logic                         tmac_ready_o,
    output logic           [NrVInsn-1:0] tmac_vinsn_done_o,
    
    // Interface with the operand queues
    input  elen_t          [1:0]         tmac_operand_i,
    input  logic           [1:0]         tmac_operand_valid_i,
    output logic           [1:0]         tmac_operand_ready_o,
    
    // Interface with the vector register file
    output logic                         tmac_result_req_o,
    output vid_t                         tmac_result_id_o,
    output vaddr_t                       tmac_result_addr_o,
    output elen_t                        tmac_result_wdata_o,
    output strb_t                        tmac_result_be_o,
    input  logic                         tmac_result_gnt_i,
    
    // Interface with the Mask unit
    input  strb_t                        mask_i,
    input  logic                         mask_valid_i,
    output logic                         mask_ready_o
  );

  import cf_math_pkg::idx_width;
  `include "common_cells/registers.svh"

  ////////////////////////////////
  //  Vector instruction queue  //
  ////////////////////////////////

  // We store a certain number of in-flight vector instructions
  localparam VInsnQueueDepth = 4; // Adjust as needed for T-MAC

  struct packed {
    vfu_operation_t [VInsnQueueDepth-1:0] vinsn;

    // Instruction execution phases pointers
    logic [idx_width(VInsnQueueDepth)-1:0] accept_pnt;
    logic [idx_width(VInsnQueueDepth)-1:0] issue_pnt;
    logic [idx_width(VInsnQueueDepth)-1:0] commit_pnt;

    // Instruction counters
    logic [idx_width(VInsnQueueDepth):0] issue_cnt;
    logic [idx_width(VInsnQueueDepth):0] commit_cnt;
  } vinsn_queue_d, vinsn_queue_q;

  // Is the vector instruction queue full?
  logic vinsn_queue_full;
  assign vinsn_queue_full = (vinsn_queue_q.commit_cnt == VInsnQueueDepth);

  // Do we have a vector instruction ready to be issued?
  vfu_operation_t vinsn_issue_d, vinsn_issue_q;
  logic           vinsn_issue_valid;
  assign vinsn_issue_d     = vinsn_queue_d.vinsn[vinsn_queue_d.issue_pnt];
  assign vinsn_issue_valid = (vinsn_queue_q.issue_cnt != '0);

  // Do we have a vector instruction with results being committed?
  vfu_operation_t vinsn_commit;
  logic           vinsn_commit_valid;
  assign vinsn_commit       = vinsn_queue_q.vinsn[vinsn_queue_q.commit_pnt];
  assign vinsn_commit_valid = (vinsn_queue_q.commit_cnt != '0);

  ////////////////////
  //  Result queue  //
  ////////////////////

  localparam int unsigned ResultQueueDepth = 2;

  // Result queue structure
  typedef struct packed {
    vid_t id;
    vaddr_t addr;
    elen_t wdata;
    strb_t be;
  } payload_t;

  // Result queue
  payload_t [ResultQueueDepth-1:0]            result_queue_d, result_queue_q;
  logic     [ResultQueueDepth-1:0]            result_queue_valid_d, result_queue_valid_q;
  logic     [idx_width(ResultQueueDepth)-1:0] result_queue_write_pnt_d, result_queue_write_pnt_q;
  logic     [idx_width(ResultQueueDepth)-1:0] result_queue_read_pnt_d, result_queue_read_pnt_q;
  logic     [idx_width(ResultQueueDepth):0]   result_queue_cnt_d, result_queue_cnt_q;

  // Is the result queue full?
  logic result_queue_full;
  assign result_queue_full = (result_queue_cnt_q == ResultQueueDepth);

  ///////////////////////////
  //  T-MAC implementation //
  ///////////////////////////
  
  // States for the T-MAC operation
  typedef enum logic [2:0] {
    IDLE,
    BIT_DECOMPOSITION,   // Decompose weight matrix into one-bit matrices
    PRECOMPUTE_LUT,      // Precompute the activation matrix into LUT
    TABLE_LOOKUP,        // Perform table lookup operations
    BIT_AGGREGATION      // Aggregate the results
  } tmac_state_e;
  
  tmac_state_e tmac_state_d, tmac_state_q;
  
  // Lookup Table (LUT) storage
  // The size would depend on the group size (g)
  localparam int unsigned LUTSize = 16; // Example size, adjust as needed
  localparam int unsigned GroupSize = 4; // Group size for LUT (g)
  
  elen_t [LUTSize-1:0] lut_d, lut_q;
  
  // Weight matrices after bit-serial decomposition
  // For simplicity, we'll assume a fixed number of bits
  localparam int unsigned WeightBits = 4; // Example, adjust based on your needs
  elen_t [WeightBits-1:0] weight_matrices_d, weight_matrices_q;
  
  // Counters for T-MAC processing
  logic [2:0] current_bit_d, current_bit_q;
  logic [7:0] current_position_d, current_position_q;
  
  // Remaining elements of the current instruction in the issue phase
  vlen_t issue_cnt_d, issue_cnt_q;
  // Remaining elements of the current instruction in the commit phase
  vlen_t commit_cnt_d, commit_cnt_q;

  // How many elements are we processing this cycle?
  logic [3:0] element_cnt_buf_issue, element_cnt_buf_commit;
  logic [1:0] issue_effective_eew, commit_effective_eew;
  logic [6:0] element_cnt_issue;
  logic [6:0] element_cnt_commit;

  ///////////////
  //  Control  //
  ///////////////

  always_comb begin : p_vtmac_control
    // Maintain state
    vinsn_queue_d = vinsn_queue_q;
    issue_cnt_d   = issue_cnt_q;
    commit_cnt_d  = commit_cnt_q;

    result_queue_d           = result_queue_q;
    result_queue_valid_d     = result_queue_valid_q;
    result_queue_read_pnt_d  = result_queue_read_pnt_q;
    result_queue_write_pnt_d = result_queue_write_pnt_q;
    result_queue_cnt_d       = result_queue_cnt_q;
    
    // T-MAC specific state
    tmac_state_d         = tmac_state_q;
    lut_d                = lut_q;
    weight_matrices_d    = weight_matrices_q;
    current_bit_d        = current_bit_q;
    current_position_d   = current_position_q;

    // Inform our status to the lane controller
    tmac_ready_o      = !vinsn_queue_full;
    tmac_vinsn_done_o = '0;

    // Do not acknowledge any operands by default
    tmac_operand_ready_o = '0;
    mask_ready_o         = '0;

    // How many elements are we processing this cycle?
    issue_effective_eew = unsigned'(vinsn_issue_q.vtype.vsew[1:0]);
    element_cnt_buf_issue = 1 << (unsigned'(EW64) - issue_effective_eew);
    element_cnt_issue = {2'b0, element_cnt_buf_issue};

    commit_effective_eew = unsigned'(vinsn_commit.vtype.vsew[1:0]);
    element_cnt_buf_commit = 1 << (unsigned'(EW64) - commit_effective_eew);
    element_cnt_commit = {2'b0, element_cnt_buf_commit};

    // T-MAC state machine
    case (tmac_state_q)
      IDLE: begin
        // Only process if we have a valid instruction
        if (vinsn_issue_valid) begin
          // If this is a T-MAC instruction (assuming TMAC is defined in the instruction set)
          if (vinsn_issue_q.op == TMAC) begin
            // Initialize for T-MAC operation
            current_bit_d = '0;
            current_position_d = '0;
            tmac_state_d = BIT_DECOMPOSITION;
          end
        end
      end
      
      BIT_DECOMPOSITION: begin
        // Check if we have the operands needed
        if (tmac_operand_valid_i[1]) begin
          // Get the weight matrix
          // Implementation of bit-serial decomposition as per the paper
          if (current_bit_q < WeightBits) begin
            // Extract one-bit matrix from weight matrix (tmac_operand_i[1])
            // This is a simplified implementation - implement the actual bit extraction logic
            for (int i = 0; i < DataWidth; i++) begin
              weight_matrices_d[current_bit_q][i] = (tmac_operand_i[1][i] >> current_bit_q) & 1'b1;
            end
            current_bit_d = current_bit_q + 1;
          end else begin
            // All bits processed, move to next state
            current_bit_d = '0;
            tmac_state_d = PRECOMPUTE_LUT;
          end
          
          // Acknowledge the weight operand
          tmac_operand_ready_o[1] = 1'b1;
        end
      end
      
      PRECOMPUTE_LUT: begin
        // Check if we have the activation operand
        if (tmac_operand_valid_i[0]) begin
          // Precompute the LUT as per the algorithm in the paper
          // This is simplified - implement the actual precomputation logic
          for (int i = 0; i < LUTSize; i++) begin
            lut_d[i] = '0;
            for (int j = 0; j < GroupSize; j++) begin
              if (i & (1 << j)) begin
                // Add activation value to LUT
                lut_d[i] = lut_d[i] + tmac_operand_i[0];
              end else begin
                // Subtract activation value from LUT
                lut_d[i] = lut_d[i] - tmac_operand_i[0];
              end
            end
          end
          
          // Acknowledge the activation operand
          tmac_operand_ready_o[0] = 1'b1;
          
          // Move to next state
          tmac_state_d = TABLE_LOOKUP;
        end
      end
      
      TABLE_LOOKUP: begin
        // Perform table lookup for each bit position
        if (current_bit_q < WeightBits) begin
          // Create a result by looking up each group in the LUT
          elen_t result_temp = '0;
          
          // Process groups (simplified implementation)
          for (int i = 0; i < (DataWidth/GroupSize); i++) begin
            // Extract the group index from one-bit matrix
            logic [GroupSize-1:0] group_index;
            for (int j = 0; j < GroupSize; j++) begin
              group_index[j] = weight_matrices_q[current_bit_q][i*GroupSize+j];
            end
            

            result_temp = result_temp + (lut_q[group_index] << (current_bit_q * DataWidth/GroupSize));
          end
          
          // Store result in the result queue
          result_queue_d[result_queue_write_pnt_q].wdata = result_temp;
          result_queue_d[result_queue_write_pnt_q].addr = 
            vaddr(vinsn_issue_q.vd, NrLanes, VLEN) + current_position_q;
          result_queue_d[result_queue_write_pnt_q].id = vinsn_issue_q.id;
          result_queue_d[result_queue_write_pnt_q].be = {StrbWidth{1'b1}}; // All bytes enabled
          
          current_bit_d = current_bit_q + 1;

          if (current_bit_d >= WeightBits) begin
            // Make the result valid
            result_queue_valid_d[result_queue_write_pnt_q] = 1'b1;
            
            // Update result queue pointers
            result_queue_cnt_d += 1;
            if (result_queue_write_pnt_q == ResultQueueDepth-1)
              result_queue_write_pnt_d = 0;
            else
              result_queue_write_pnt_d = result_queue_write_pnt_q + 1;
              
            tmac_state_d = BIT_AGGREGATION;
          end
        end
      end
      
      BIT_AGGREGATION: begin
        current_position_d = current_position_q + 1;
        
        // Reset bit counter
        current_bit_d = '0;
        
        // Check if we've processed all elements
        if (current_position_d >= issue_cnt_q) begin
          // Finished processing, go back to idle
          tmac_state_d = IDLE;
          
          // Update counters
          issue_cnt_d = '0;
          
          vinsn_queue_d.issue_cnt -= 1;
          if (vinsn_queue_q.issue_pnt == VInsnQueueDepth-1)
            vinsn_queue_d.issue_pnt = '0;
          else
            vinsn_queue_d.issue_pnt = vinsn_queue_q.issue_pnt + 1;
          
          // If next instruction is ready, prepare counters
          if (vinsn_queue_d.issue_cnt != 0)
            issue_cnt_d = vinsn_queue_q.vinsn[vinsn_queue_d.issue_pnt].vl;
          
          // Move to commit phase
          commit_cnt_d = '0;
        end else begin
          // More elements to process, go back to bit decomposition
          tmac_state_d = BIT_DECOMPOSITION;
        end
      end
      
      default: tmac_state_d = IDLE;
    endcase

    //////////////////////////////////
    //  Write results into the VRF  //
    //////////////////////////////////

    tmac_result_wdata_o = result_queue_q[result_queue_read_pnt_q].wdata;
    tmac_result_req_o = result_queue_valid_q[result_queue_read_pnt_q];
    tmac_result_addr_o = result_queue_q[result_queue_read_pnt_q].addr;
    tmac_result_id_o   = result_queue_q[result_queue_read_pnt_q].id;
    tmac_result_be_o   = result_queue_q[result_queue_read_pnt_q].be;

    // Received a grant from the VRF.
    // Deactivate the request.
    if (tmac_result_gnt_i) begin
      result_queue_valid_d[result_queue_read_pnt_q] = 1'b0;
      result_queue_d[result_queue_read_pnt_q]       = '0;

      // Increment the read pointer
      if (result_queue_read_pnt_q == ResultQueueDepth-1) 
        result_queue_read_pnt_d = 0;
      else 
        result_queue_read_pnt_d = result_queue_read_pnt_q + 1;

      // Decrement the counter of results waiting to be written
      result_queue_cnt_d -= 1;
    end

    // Finished committing the results of a vector instruction
    if (vinsn_commit_valid && (commit_cnt_d == '0)) begin
      // Mark the vector instruction as being done
      tmac_vinsn_done_o[vinsn_commit.id] = 1'b1;

      // Update the commit counters and pointers
      vinsn_queue_d.commit_cnt -= 1;
      if (vinsn_queue_d.commit_pnt == VInsnQueueDepth-1) 
        vinsn_queue_d.commit_pnt = '0;
      else 
        vinsn_queue_d.commit_pnt += 1;

      // Update the commit counter for the next instruction
      if (vinsn_queue_d.commit_cnt != '0)
        commit_cnt_d = vinsn_queue_q.vinsn[vinsn_queue_d.commit_pnt].vl;
    end

    //////////////////////////////
    //  Accept new instruction  //
    //////////////////////////////

    if (!vinsn_queue_full && vfu_operation_valid_i && vfu_operation_i.op == TMAC) begin
      vinsn_queue_d.vinsn[vinsn_queue_q.accept_pnt] = vfu_operation_i;

      // Initialize counters if the instruction queue was empty
      if (vinsn_queue_d.issue_cnt == '0) begin
        issue_cnt_d = vfu_operation_i.vl;
      end
      
      if (vinsn_queue_d.commit_cnt == '0)
        commit_cnt_d = vfu_operation_i.vl;

      // Bump pointers and counters of the vector instruction queue
      vinsn_queue_d.accept_pnt += 1;
      vinsn_queue_d.issue_cnt += 1;
      vinsn_queue_d.commit_cnt += 1;
    end
  end : p_vtmac_control

  // Register the state
  always_ff @(posedge clk_i or negedge rst_ni) begin : p_vtmac_ff
    if (!rst_ni) begin
      vinsn_queue_q         <= '0;
      vinsn_issue_q         <= '0;
      
      result_queue_q           <= '0;
      result_queue_valid_q     <= '0;
      result_queue_write_pnt_q <= '0;
      result_queue_read_pnt_q  <= '0;
      result_queue_cnt_q       <= '0;
      
      issue_cnt_q         <= '0;
      commit_cnt_q        <= '0;
      tmac_state_q        <= IDLE;
      lut_q               <= '0;
      weight_matrices_q   <= '0;
      current_bit_q       <= '0;
      current_position_q  <= '0;
    end else begin
      vinsn_queue_q         <= vinsn_queue_d;
      vinsn_issue_q         <= vinsn_issue_d;
      
      result_queue_q           <= result_queue_d;
      result_queue_valid_q     <= result_queue_valid_d;
      result_queue_write_pnt_q <= result_queue_write_pnt_d;
      result_queue_read_pnt_q  <= result_queue_read_pnt_d;
      result_queue_cnt_q       <= result_queue_cnt_d;
      
      issue_cnt_q         <= issue_cnt_d;
      commit_cnt_q        <= commit_cnt_d;
      tmac_state_q        <= tmac_state_d;
      lut_q               <= lut_d;
      weight_matrices_q   <= weight_matrices_d;
      current_bit_q       <= current_bit_d;
      current_position_q  <= current_position_d;
    end
  end : p_vtmac_ff

endmodule : vtmac