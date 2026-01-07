library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity tb_MAC is
end tb_MAC;

architecture Behavioral of tb_MAC is

    signal clk   : std_logic := '0';
    signal reset : std_logic := '0';
    signal en    : std_logic := '0';
    signal a     : signed(7 downto 0) := (others => '0');
    signal b     : signed(7 downto 0) := (others => '0');
    signal acc   : signed(15 downto 0);

    constant clk_period : time := 10 ns;

begin

   
    DUT: entity work.MAC
        port map (
            clk     => clk,
            reset   => reset,
            en      => en,
            a       => a,
            b       => b,
            acc_reg => acc
        );

   
    clk_process : process
    begin
        clk <= '0';
        wait for clk_period / 2;
        clk <= '1';
        wait for clk_period / 2;
    end process;

   
    stim_proc : process
    begin
        reset <= '1';
        wait for clk_period;
        reset <= '0';

        en <= '1';
        a  <= to_signed(3, 8);
        b  <= to_signed(4, 8);
        wait for clk_period;

        a <= to_signed(-2, 8);
        b <= to_signed(5, 8);
        wait for clk_period;

        en <= '0';
        a  <= to_signed(10, 8);
        b  <= to_signed(10, 8);
        wait for clk_period;

        en <= '1';
        a  <= to_signed(1, 8);
        b  <= to_signed(6, 8);
        wait for clk_period;

        wait;
    end process;

end Behavioral;
