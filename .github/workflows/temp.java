import java.util.*;

public class temp {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a character: ");
        char ch = sc.next().charAt(0);

        int ascii = (int) ch;
        String binary = String.format("%8s", Integer.toBinaryString(ascii)).replace(' ', '0');

        int count = 0;
        for (char bit : binary.toCharArray()) {
            if (bit == '1') count++;
        }
        int parityBit = (count % 2 == 0) ? 0 : 1;

        System.out.println("Original ASCII: " + ascii);
        System.out.println("Binary (8 bits): " + binary);
        System.out.println("Parity bit (Even parity): " + parityBit);

        String transmitted = binary + parityBit;
        System.out.println("Transmitted bits: " + transmitted);

        System.out.print("Enter received bits (9 bits): ");
        String received = sc.next();

        String dataBits = received.substring(0, 8);
        int receivedParity = Character.getNumericValue(received.charAt(8));

        int ones = 0;
        for (char bit : dataBits.toCharArray()) {
            if (bit == '1') ones++;
        }

        int calculatedParity = (ones % 2 == 0) ? 0 : 1;

        if (calculatedParity == receivedParity) {
            int correctedAscii = Integer.parseInt(dataBits, 2);
            System.out.println("No Error Detected!");
            System.out.println("Received Character: " + (char) correctedAscii);
        } else {
            System.out.println("Error Detected in Transmission!");
        }
    }
}
